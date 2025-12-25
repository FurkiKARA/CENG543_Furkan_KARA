print("Checking run_sbert.txt...")
try:
    with open("run_sbert.txt", "r") as f:
        lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        print("First 5 lines:")
        for l in lines[:5]:
            print(repr(l)) # repr shows hidden characters like \t or \n
except FileNotFoundError:
    print("File not found!")