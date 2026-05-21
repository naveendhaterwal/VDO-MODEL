import requests
import time
import sys

API_URL = "http://localhost:8000"

def test_flow():
    print(f"Submitting mock video generation request to {API_URL}/generate...")
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={"prompt": "A cinematic video of a mock test running end-to-end"}
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to submit request: {e}")
        sys.exit(1)

    data = response.json()
    job_id = data["job_id"]
    print(f"✅ Job successfully queued! Job ID: {job_id}")

    print("Polling job status...")
    while True:
        try:
            res = requests.get(f"{API_URL}/status/{job_id}")
            res.raise_for_status()
            status_data = res.json()
            status = status_data["status"]
            print(f"Status: {status}")
            
            if status == "finished":
                print(f"✅ Video generated successfully! Result URL: {status_data['result']}")
                break
            elif status == "failed":
                print(f"❌ Job failed: {status_data.get('result')}")
                break
                
            time.sleep(2)
        except Exception as e:
            print(f"Failed to get status: {e}")
            time.sleep(2)

if __name__ == "__main__":
    test_flow()
