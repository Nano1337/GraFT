import socket
import sys
import json
import argparse
import time


class Client():
    def __init__(self, socket_num=1000):
        self.socket_num = socket_num

    def submit_job(self, script):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', self.socket_num))
        client_socket.send(script.encode('utf-8'))
        response = client_socket.recv(1024).decode('utf-8')

        if response.startswith('{'):  # if response is a JSON (i.e., it's an error message)
            response_data = json.loads(response)
            return f"Error: {response_data.get('error')}"
        else:
            job_id = response.split(": ")[-1]  # get the job id from the response
            return f"{response}"

        client_socket.close()

    def kill_job(self, screen_name):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', self.socket_num))
        client_socket.send(f"kill:{screen_name}".encode('utf-8'))
        client_socket.close()
        return f"Killing job with ID '{screen_name}'..."

    def kill_all_jobs(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', self.socket_num))
        client_socket.send("kill_all".encode('utf-8'))
        client_socket.close()
        return "Killing all jobs..."

    def get_job_status(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', self.socket_num))
        client_socket.send("get_status".encode('utf-8'))
        job_status = client_socket.recv(1024).decode('utf-8')
        client_socket.close()

        if job_status:
            print(job_status, "job_status")
            job_status = json.loads(job_status)
            if isinstance(job_status, dict) and 'message' in job_status:
                return job_status['message']
            elif job_status:
                print("Jobs status:")
                output = ""
                for job in job_status:
                    output += f"Job ID: {job['job_id']}, Status: {job['status']}, Script: {job['script']}, Timestamp: {job['timestamp']}"
                    output += "<br>"
                return output
        else:
            return "Failed to retrieve job status."

    def get_job_queue(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', self.socket_num))
        client_socket.send("get_queue".encode('utf-8'))
        queue = client_socket.recv(1024).decode('utf-8')
        client_socket.close()

        if queue:
            queue = json.loads(queue)
            if isinstance(queue, dict) and 'error' in queue:  # If an error message was received
                print(f"Error: {queue.get('error')}")
            else:
                print("Job queue:")
                for job in queue:
                    print(f"Job ID: {job['job_id']}, Status: {job['status']}, Script: {job['script']}")
        else:
            print("Job queue is empty.")

def print_menu():
    print("\n==== MENU ====")
    print("1. Submit Job")
    print("2. Kill Job")
    print("3. Kill All Jobs")
    print("4. Get Job Status")
    print("5. Get Job Queue")
    print("0. Exit")

def process_choice(choice):
    if choice == "":
        return False
    try:
        if choice == "1":
            script = ""
            while not script:
                script = input("Enter the path to the bash script: ")
            submit_job(script)
        elif choice == "2":
            screen_name = input("Enter the screen name of the job to kill: ")
            kill_job(screen_name)
        elif choice == "3":
            kill_all_jobs()
        elif choice == "4":
            get_job_status()
        elif choice == "5":
            get_job_queue()
        elif choice == "0":
            sys.exit()
        else:
            print("Invalid choice. Please try again.")
        return False
    except Exception as e:
        print(str(e))
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Scheduler')
    parser.add_argument("-p", "--port", type=int, default=10000, help="Port number")
    args = parser.parse_args()
    # global self.socket_num 
    # self.socket_num = args.port

    while True:
        print_menu()
        choice = input("Enter your choice (0-5): ")
        if not process_choice(choice):
            print()
            input("Press enter to continue...")  # Add a pause so user can read the results
