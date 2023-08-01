import socket
import threading
import subprocess
import os
import time
from collections import deque
from pynvml import *
import uuid
import json
import atexit
import argparse
import time

class GPUScheduler:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.job_queue = deque()  
        self.oom_job_queue = deque()
        self.non_oom_job_finished = False  # Flag to indicate if a non-OOM job has finished

        self.start_time = time.time()
        self.error_logs = ""

        self.gpu_states = [None] * num_gpus 
        self.running_jobs = []  # Added to keep track of running jobs
        self.completed_jobs = []  # Added to keep track of completed jobs
        nvmlInit()
        self.gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]
        self.server_socket_thread = None
        atexit.register(self.on_exit)

    def on_exit(self):
        print("Server is stopping, killing all jobs...")
        self.kill_all_jobs()


    def run_job(self, script, job_id):
        screen_name = f"job_{job_id}" 
        args = ['/usr/bin/screen', '-dm', '-S', screen_name, 'sh', '-c', f'sh ./{script} > error_logs/{screen_name}.log 2>&1; echo $? > error_logs/{screen_name}.exit'] 
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(self.num_gpus)))

        if not os.path.exists(script):
            return {'error': f"Script path '{script}' not found."}

        try:
            process = subprocess.Popen(args, env=env, stderr=subprocess.PIPE, 
                                   universal_newlines=True, preexec_fn=os.setsid)
        

            attach_command = f"screen -r {screen_name}"

            timestamp = self.get_elapsed_time(time.time())
            print(f"Started job on screen {screen_name} with process group {os.getpgid(process.pid)}. To attach, use '{attach_command}'")
            return {'process': process, 'screen_name': screen_name, 'script': script, 'timestamp': timestamp, "log_file": f"error_logs/{screen_name}.log"}

        except Exception as e:
            return {'error': str(e)}


    def get_elapsed_time(self, timestamp): 
        total_time = int(timestamp - self.start_time)
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = ((total_time % 3600) % 60) % 60

        elapsed_time = f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}"
        return elapsed_time


    def schedule_next_job(self):
        if len(self.running_jobs) == 0:
            self.non_oom_job_finished = True

        if not self.job_queue and not (self.oom_job_queue and self.non_oom_job_finished):  # If both queues are empty or the OOM queue is only available but no non-OOM job has finished
            return

        if self.job_queue:  # If there's a job in the normal queue
            script_info = self.job_queue.popleft()

        elif self.non_oom_job_finished:  # If the normal queue is empty and a non-OOM job has finished, take a job from the OOM queue
            script_info = self.oom_job_queue.popleft()
            self.non_oom_job_finished = False  # Reset the flag

        process_info = self.run_job(script_info['script'], script_info['job_id'])

        if 'error' in process_info:
            print(f"Error when starting job {script_info['script']}: {process_info['error']}")
            # Just skip this job and move to the next one
        else:
            script_info.update(process_info)
            script_info['status'] = 'running'
            self.running_jobs.append(script_info)


    def is_screen_running(self, screen_name):
        cmd = ['/usr/bin/screen', '-ls', screen_name]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        return screen_name in stdout.decode()

    def update_job_status(self):
        for process_info in self.running_jobs:  # If GPU is busy
            process = process_info['process'] # Get the process object
            
            if process.poll() is not None and not self.is_screen_running(process_info["screen_name"]) and process_info['status'] != "killed":  # If the process has finished

                # get exit code
                # print(process.returncode)
                # if process.returncode != None: 
                #     exit_code = process.returncode
                # else:
                with open(f"error_logs/{process_info['screen_name']}.log", 'r') as file:
                    log_content = file.read()

                if 'CUD' in log_content:  # If the process had OutOfMemoryError
                    # Move the job to the OOM queue
                    self.oom_job_queue.append(process_info)
                    process_info['status'] = 'requeued until memory is available'
                    process_info['error'] = 'Job was requeued due to OutOfMemoryError.'
                    self.non_oom_job_finished = False  # Reset the flag
                    os.remove(f"error_logs/{process_info['screen_name']}.log")
                    os.remove(f"error_logs/{process_info['screen_name']}.exit")
                    process_info['log_file'] = None
                    self.running_jobs.remove(process_info)

                else:
                    # If a non-OOM job has finished
                    self.non_oom_job_finished = True

                    with open(f"error_logs/{process_info['screen_name']}.exit", 'r') as file:
                        exit_code = int(file.read().strip())

                    if exit_code != 0:  # If the process failed
                        process_info['status'] = 'failed'
                        process_info['error'] = 'Job failed with exit code: ' + str(exit_code)
                        # Dump the output to a log file
                        print(f"Job {process_info['screen_name']} failed with exit code {exit_code}. See error_logs/{process_info['screen_name']}.log for details.")
                        self.error_logs += f"Job {process_info['screen_name']} failed with exit code {exit_code}. See error_logs/{process_info['screen_name']}.log for details.<br>"

                    else:  # If the process finished successfully
                        process_info['status'] = 'completed'
                        os.remove(f"error_logs/{process_info['screen_name']}.log")
                        process_info['log_file'] = None

                    os.remove(f"error_logs/{process_info['screen_name']}.exit")
                    self.completed_jobs.append(process_info)  # Add process info to completed jobs
                    self.running_jobs.remove(process_info)  # Remove process info from running jobs
            

    def monitor_jobs(self):
        while True:
            self.schedule_next_job()
            self.update_job_status()
            time.sleep(1)  # Sleep for a short time

    def start_server_socket(self, host, port):
        if self.server_socket_thread is not None:
            return {'error': 'Server socket is already running.'}

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, port))
        server.listen()
        print(f"Server activated and listening on {host}:{port}")

        self.server_socket_thread = threading.Thread(target=self.handle_client, args=(server,))
        self.server_socket_thread.start()

    def handle_client(self, server):
    
        while True:
            client_socket, _ = server.accept()
            request = client_socket.recv(1024).decode('utf-8')
                

            if request.startswith("kill:"):
                screen_name = request[5:]
                self.kill_job(screen_name)
                client_socket.send(f"Job {screen_name} was killed.".encode('utf-8'))
            elif request == "kill_all":
                self.kill_all_jobs()
                client_socket.send("All jobs were killed.".encode('utf-8'))
            elif request == "get_status":
                self.send_job_status(client_socket)
            elif request == "get_queue":
                self.send_job_queue(client_socket)
            elif request.startswith("get_job_status:"):
                job_id = request[len("get_job_status:"):]
                job_status = self.get_job_status(job_id)
                client_socket.send(json.dumps(job_status).encode('utf-8'))
            else:  # Job submission case
                if not os.path.exists(request):
                    client_socket.send(json.dumps({'error': f"Script path '{request }' not found."}).encode('utf-8'))
                    continue

                
                job_id = str(uuid.uuid4())[:8]
                script_info = {'script': request, 'job_id': job_id, 'status': 'queued'}
                self.job_queue.append(script_info)
                client_socket.send(f'Queued with Job ID: {job_id}'.encode('utf-8'))


            client_socket.close()

    def kill_job(self, screen_name):
        job_found = False
        for process_info in list(self.running_jobs) + list(self.job_queue) + list(self.oom_job_queue):
            if process_info['screen_name'] == "job_" + screen_name:
                if process_info in self.running_jobs:
                    try:
                        # Sending SIGTERM to all processes running within the screen session
                        subprocess.check_output(['screen', '-S', screen_name, '-X', 'quit'])
                        print(f"Job {screen_name} was killed.")
                        job_found = True
                    except subprocess.CalledProcessError:
                        print(f"Failed to kill job {screen_name}")
                    
                    self.running_jobs.remove(process_info)
                    self.non_oom_job_finished = True

                elif process_info in self.job_queue:
                    self.job_queue.remove(process_info)

                elif process_info in self.oom_job_queue:
                    self.oom_job_queue.remove(process_info)

                process_info['status'] = 'killed'
                self.completed_jobs.append(process_info)

                # remove any exit and log files if they exist
                if os.path.exists(f"error_logs/{process_info['screen_name']}.log"):
                    os.remove(f"error_logs/{process_info['screen_name']}.log")

                if os.path.exists(f"error_logs/{process_info['screen_name']}.exit"):
                    os.remove(f"error_logs/{process_info['screen_name']}.exit")
                
                process_info['log_file'] = None
                
        if not job_found:
            print(f"No job found with screen name: {screen_name}")

    def kill_all_jobs(self):
        self.job_queue.clear()  # Clear all queued jobs
        for process_info in self.running_jobs:
            try:
                # Sending SIGTERM to all processes running within the screen session
                subprocess.check_output(['screen', '-S', process_info['screen_name'], '-X', 'quit'])
                print(f"Job {process_info['screen_name']} was killed.")
            except subprocess.CalledProcessError:
                print(f"Failed to kill job {process_info['screen_name']}")
            os.remove(f"error_logs/{process_info['screen_name']}.log")
            process_info['status'] = 'killed'
            self.running_jobs.remove(process_info)
            self.completed_jobs.append(process_info)

            # remove any exit and log files if they exist
            if os.path.exists(f"error_logs/{process_info['screen_name']}.log"):
                os.remove(f"error_logs/{process_info['screen_name']}.log")

            if os.path.exists(f"error_logs/{process_info['screen_name']}.exit"):
                os.remove(f"error_logs/{process_info['screen_name']}.exit")

            process_info['log_file'] = None

    def send_job_status(self, client_socket):
        job_status = []
        for process_info in self.running_jobs:
            job_status.append(self.sanitize_process_info(process_info))
        
        # For jobs in the queue, sanitize before appending
        for job in self.job_queue:
            job_status.append(self.sanitize_process_info(job))
            
        # For completed jobs, sanitize before appending
        for job in self.completed_jobs:
            job_status.append(self.sanitize_process_info(job))

        if not job_status:  # If no jobs are currently running or queued
            job_status = {'message': 'No jobs currently running or queued.'}
        
        client_socket.send(json.dumps(job_status).encode('utf-8'))


    def sanitize_process_info(self, process_info):
        # Create a copy of process_info and remove the 'process' key before returning
        sanitized_info = process_info.copy()
        sanitized_info.pop('process', None)
        return sanitized_info

    def send_job_queue(self, client_socket):
        queue_status = []
        for job in self.job_queue:
            queue_status.append(job)
        client_socket.send(json.dumps(queue_status).encode('utf-8'))

def main(socket_num):
    parser = argparse.ArgumentParser(description='GPU Scheduler')
    parser.add_argument("-p", "--port", type=int, default=socket_num, help="Port number")
    args = parser.parse_args()

    if not os.path.exists('error_logs'):
        os.makedirs('error_logs')

    scheduler = GPUScheduler(num_gpus=8)
    threading.Thread(target=scheduler.monitor_jobs).start()
    scheduler.start_server_socket('localhost', args.port)
    return scheduler

if __name__ == "__main__":
    main()
