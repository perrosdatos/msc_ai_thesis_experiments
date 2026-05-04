import os
import shutil

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Clean CSV data
    csv_dir = os.path.join(base_dir, "csv_data")
    if os.path.exists(csv_dir):
        for f in os.listdir(csv_dir):
            if f.endswith(".csv"):
                file_path = os.path.join(csv_dir, f)
                os.remove(file_path)
                print(f"🗑️ Removed: {file_path}")
                
    # 2. Clean root HTML reports
    html_dir = os.path.join(base_dir, "html_reports")
    if os.path.exists(html_dir):
        for f in os.listdir(html_dir):
            file_path = os.path.join(html_dir, f)
            if os.path.isfile(file_path) and f.endswith(".html"):
                os.remove(file_path)
                print(f"🗑️ Removed: {file_path}")
                
        # 3. Clean subfolder sweep_dashboards
        sweep_dashboards = os.path.join(html_dir, "sweep_dashboards")
        if os.path.exists(sweep_dashboards):
            shutil.rmtree(sweep_dashboards)
            print(f"🗑️ Removed directory and contents: {sweep_dashboards}")
            
    print("\\n✅ Cleanup complete! Environment is ready for a fresh sweep.")

if __name__ == "__main__":
    main()
