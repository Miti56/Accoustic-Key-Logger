import subprocess
import sys
import time


def display_prompt(prompt):
    print(prompt, end=' ')
    return input()

def run_record():
    try:
        subprocess.run(['python3', 'record/record.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")

def run_data():
    try:
        subprocess.run(['python3', 'DataVisualisation/dataVisualisation.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")

def run_modelCC():
    try:
        subprocess.run(['python3', 'model/modelCC.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")

def run_modelML():
    try:
        subprocess.run(['python3', 'model/modelML.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")

def display_additional_information():
    print("Additional Information:")
    print("- Cross-Correlation Model: This model uses cross-correlation to compare the input audio signals with known "
          "patterns.")
    print("It calculates the similarity between the input signals and the patterns, helping to identify specific "
          "patterns in the audio.")
    print("- Neural Network Model: This model utilizes a neural network architecture to analyze the audio signals.")
    print("It learns patterns and features from the input data through training and can classify audio based on the "
          "learned patterns.")

def main():
    print("Welcome to the UI!")

    q1 = display_prompt("Are you in a quiet space to start the recording? ").upper() == 'Y'

    if q1:
        # Run the original program
        run_data()
    else:
        q2 = display_prompt("Do you wish to continue with already existent data? ").upper() == 'Y'
        if q2:
            pass
        else:
            print("Program terminated.")
            sys.exit()
        print("Redirecting...")
        time.sleep(3)


    q3 = display_prompt("Visualise Data? ").upper() == 'Y'

    if q3:
        # Run the original program
        run_data()
        print("Data analysis performed.")
        q4 = display_prompt("Are the results satisfactory? ").upper() == 'Y'
        if q4:
            pass
        else:
            print("Program terminated. Try again in a quieter environment")
            sys.exit()

    else:
        print("Redirecting to the model creation...")
        time.sleep(3)

    q5 = display_prompt("Two models are currently available: Cross-Correlation or Neural Network (Press H for "
                        "additional information").upper() == 'Y'

    if q5 == 'C':
        # Perform Cross-Correlation
        run_modelCC()
    elif q5 == 'N':
        # Perform Neural Network
        run_modelML()
    elif q5 == 'H':
        # Display additional information
        display_additional_information()
    else:
        # Invalid input
        print("Invalid input. Please select a valid option.")


if __name__ == "__main__":
    main()

