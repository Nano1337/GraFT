from flask import Flask, render_template, request, jsonify, send_file
import os
import activate_server 
import subprocess
import activate_client

app = Flask(__name__)
# activate_server.main()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/launch_server', methods=['POST'])
def launch_server():
    global server
    global client
    try: 
        port = int(request.form.get('port'))
        server = activate_server.main(port)
        client = activate_client.Client(port)
    except:
        return "Invalid port number."
    return "Server launched"

@app.route('/submit_job', methods=['POST'])
def submit_job():
    script_path = request.form.get('script')
    if script_path == "":
        return "No script path provided."
    # Perform job submission logic here

    return client.submit_job(script_path)
    

@app.route('/kill_job', methods=['POST'])
def kill_job():
    if client is None:
        return "Server not connected."
    screen_name = request.form.get('screen_name')
    # Perform job killing logic here
    kill_output = client.kill_job(screen_name)
    # Replace the following line with your actual implementation
    return kill_output

@app.route('/kill_all_jobs', methods=['POST'])
def kill_all_jobs():
    if client is None:
        return "Server not connected."
    # Perform logic to kill all jobs here
    kill_all_output = client.kill_all_jobs()
    # Replace the following line with your actual implementation
    return kill_all_output

@app.route('/get_job_status', methods=['GET'])
def get_job_status():
    # Perform logic to retrieve job status here
    #output = client.get_job_status()
    # Replace the following line with your actual implementation
    #job_status = "Job status: Running"
    if server is None:
        return []

    job_status = []
    # For jobs in the queue, sanitize before appending
    for job in server.running_jobs:
        sanitized_info = job.copy()
        sanitized_info.pop('process', None)
        job_status.insert(0, sanitized_info)
    
    # For jobs in the queue, sanitize before appending
    for job in server.job_queue:
        sanitized_info = job.copy()
        sanitized_info.pop('process', None)
        job_status.append(sanitized_info)
        
    # For completed jobs, sanitize before appending
    for job in server.completed_jobs:
        sanitized_info = job.copy()
        sanitized_info.pop('process', None)
        job_status.append(sanitized_info)

    # For OOM jobs, sanitize before appending
    for job in server.oom_job_queue:
        sanitized_info = job.copy()
        sanitized_info.pop('process', None)
        job_status.append(sanitized_info)

    return jsonify(job_status)

@app.route('/check_error_log', methods=['POST'])
def check_error_log():
    # Perform logic to check error log here
    # Replace the following line with your actual implementation
    error_log = server.error_logs
    if error_log == "":
        return "No errors found."
    return error_log


@app.route('/log-file/<path:filename>')
def serve_log_file(filename):
    # make sure you secure this if filename can come from user input
    return send_file(f'../{filename}', mimetype='text/plain')

@app.route('/nvidia-smi', methods=['GET'])
def get_nvidia_smi():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')


if __name__ == '__main__':
    client = None
    server = None
    app.run(port=0)