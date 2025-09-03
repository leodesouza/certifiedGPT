
import subprocess

def mount_gcsfuse(bucket_name="certifiedgpt_storage", mount_point="~/storage"):    
    try:
        subprocess.run(["gcsfuse", bucket_name, mount_point], check=True)
        print(f"Successfully mounted {bucket_name} at {mount_point}")        
    except subprocess.CalledProcessError as e:        
        pass

# Run the function

if __name__ == "__main__":
    mount_gcsfuse()
