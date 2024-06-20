import os

def rename_files_in_directory(directory, prefix="0000dfgdferwerwDF"):
    try:
        # List all files in the given directory
        files = os.listdir(directory)
        
        count = 1
        
        for filename in files:
            # Construct the full file path
            old_file_path = os.path.join(directory, filename)
            
            # Check if it's a file
            if os.path.isfile(old_file_path):
                # Create the new filename with the prefix and count
                name, ext = os.path.splitext(filename)
                new_filename = f"{prefix}{count}{ext}"
                new_file_path = os.path.join(directory, new_filename)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_filename}'")
                
                count += 1
        
        print("Renaming completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
directory_path = r"C:\Users\amanda\Desktop\bottle classification types\temp"
rename_files_in_directory(directory_path)
