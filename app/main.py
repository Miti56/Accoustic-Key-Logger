import subprocess
import sys
import time


class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


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


def run_modelVisualisation():
    try:
        subprocess.run(['python3', 'DataVisualisation/modelVisualisation.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")


def run_testLiveModel():
    try:
        subprocess.run(['python3', 'test/liveModel.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")


def run_testModelCC():
    try:
        subprocess.run(['python3', 'test/modelCCTest.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")


def run_testModelML():
    try:
        subprocess.run(['python3', 'test/modelMLTest.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")


def run_testLong():
    try:
        subprocess.run(['python3', 'test/testLong.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")


def run_fakeKeys():
    try:
        subprocess.run(['python3', 'utils/keyboardSoundEmulation.py'], check=True)
    except subprocess.CalledProcessError:
        print("Error running the original program.")


def display_additional_information():
    print(f"{Color.HEADER}======================")
    print("Additional Information:")
    print("----------------------")
    print("- Cross-Correlation Model: This model uses cross-correlation to compare the input audio signals with known "
          "patterns.")
    print("  It calculates the similarity between the input signals and the patterns, helping to identify specific "
          "patterns in the audio.")
    print("- Neural Network Model: This model utilizes a neural network architecture to analyze the audio signals.")
    print("  It learns patterns and features from the input data through training and can classify audio based on the "
          "learned patterns.")
    print(f"======================{Color.END}")


def display_additional_information2():
    print(f"{Color.HEADER}======================")
    print("Additional Information:")
    print("----------------------")
    print("- Cross-Correlation Model: This model uses cross-correlation to compare the input audio signals with known "
          "patterns.")
    print("  It calculates the similarity between the input signals and the patterns, helping to identify specific "
          "patterns in the audio.")
    print("- Neural Network Model: This model utilizes a neural network architecture to analyze the audio signals.")
    print("  It learns patterns and features from the input data through training and can classify audio based on the "
          "learned patterns.")
    print(f"======================{Color.END}")


def main():
    print(f"{Color.BLUE}====================")
    print("Welcome to the Keyboard Listening App!")
    print("This app allows you to analyze and classify audio recordings of keyboard sounds.")
    print("Here is a brief overview of the steps involved:")

    print(f"{Color.BLUE}1. Recording:")
    print("   - You will be prompted to ensure you are in a quiet space to start the recording.")
    print("   - If you have already recorded audio files, you can choose to continue with them.")
    print("   - Otherwise, you can use the provided 'record.py' tool to record keyboard sounds.")

    print(f"{Color.BLUE}2. Data Visualization:")
    print("   - You will have the option to visualize the recorded data.")
    print("   - The 'DataVisualisation' module will be used to perform data analysis.")

    print(f"{Color.BLUE}3. Model Creation:")
    print("   - You will be asked to choose between two models: Cross-Correlation (CC) or Neural Network (NN).")
    print("   - The chosen model will be trained using the recorded data.")
    print("   - The 'modelCC.py' and 'modelML.py' modules will be used to create and train the models.")

    print(f"{Color.BLUE}4. Model Visualization:")
    print("   - You will have the option to visualize the trained model.")
    print("   - The 'modelVisualisation' module will be used to analyze and visualize the model.")

    print(f"{Color.BLUE}5. Testing:")
    print("   - You will have the option to perform different tests on the model.")
    print(
        "- Tests can be conducted using the 'liveModel.py', 'modelCCTest.py', 'modelMLTest.py', or 'testLong.py' "
        "modules.")

    print(f"{Color.BLUE}====================")
    print("Welcome!")

    q1 = display_prompt("Are you in a quiet space to start the recording? ").upper() == 'Y'

    if q1:
        print(f"{Color.GREEN}====================")
        # Run the original program
        run_data()
    else:
        q2 = display_prompt("Do you wish to continue with already existent data? ").upper() == 'Y'
        if q2:
            pass
        else:
            print("Program terminated.")
            sys.exit()
        print(f"{Color.YELLOW}====================")
        print("Redirecting...")
        time.sleep(3)

    q3 = display_prompt("Visualise Data? ").upper() == 'Y'

    if q3:
        print(f"{Color.GREEN}====================")
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
        print(f"{Color.YELLOW}====================")
        print("Redirecting to the model creation...")
        time.sleep(3)

    while True:
        q5 = display_prompt("Two models are currently available: Cross-Correlation (CC) or Neural Network (NN "
                            "(Press H for additional information)").upper()

        if q5 == 'CC':
            print(f"{Color.GREEN}====================")
            # Perform Cross-Correlation
            run_modelCC()
            break
        elif q5 == 'NN':
            print(f"{Color.GREEN}====================")
            # Perform Neural Network
            run_modelML()
            break
        elif q5 == 'H':
            print(f"{Color.HEADER}====================")
            # Display additional information
            display_additional_information()
        else:
            # Invalid input
            print("Invalid input. Please select a valid option.")

    print(f"{Color.GREEN}====================")
    print("Model has been trained!")
    print("Redirecting to the model visualisation...")
    time.sleep(3)

    q6 = display_prompt("Visualise Model? ").upper() == 'Y'

    if q6:
        print(f"{Color.GREEN}====================")
        # Run the original program
        run_modelVisualisation()
        print("Model analysis performed.")
        q4 = display_prompt("Are the results satisfactory? ").upper() == 'Y'
        if q4:
            pass
        else:
            print("Program terminated. Try again in a quieter environment")
            sys.exit()

    else:
        print(f"{Color.YELLOW}====================")
        print("Redirecting to the testing suite...")
        time.sleep(3)

    while True:
        print(f"{Color.BLUE}====================")
        print("Which test do you wish to perform?")
        print("Press H for a description of the different tests.")
        print("Press Q to end the tests")
        q7 = display_prompt("Test using: Live Model (LM), ML Model (ML), CC Model (CC) or Wav File Test (WT)?").upper()

        if q7 == 'LM':
            print(f"{Color.GREEN}====================")
            run_testLiveModel()
        elif q7 == 'ML':
            print(f"{Color.GREEN}====================")
            run_testModelML()
        elif q7 == 'CC':
            print(f"{Color.GREEN}====================")
            run_testModelCC()
        elif q7 == 'WT':
            print(f"{Color.GREEN}====================")
            run_testLong()
        elif q7 == 'H':
            print(f"{Color.HEADER}====================")
            # Display additional information
            display_additional_information2()
        elif q7 == 'Q':
            break

        else:
            # Invalid input
            print("Invalid input. Please select a valid option.")

    print("The program has finished running, if you wish, you can replace your keystroke sounds by fake ones ("
          "Press H to learn more)")

    q8 = display_prompt("Run the Keyboard replacer sound").upper() == 'Y'

    if q8:
        print(f"{Color.GREEN}====================")
        # Run the original program
        run_fakeKeys()
        print("Model analysis performed.")

    else:
        print(f"{Color.YELLOW}====================")
        print("Ending the program...")
        time.sleep(3)


if __name__ == "__main__":
    main()
