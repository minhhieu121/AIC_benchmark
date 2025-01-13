import re

def check_matches(input_text):
    # Define regex pattern to extract Result and Label
    pattern = r'Resutl: (video\d+) - Label: (video\d+)'
    
    # Find all matches in input text
    matches = re.finditer(pattern, input_text)
    
    # Initialize counters
    correct = 0
    total = 0
    
    # Process each match
    for match in matches:
        total += 1
        result, label = match.groups()
        if result == label:
            correct += 1
    
    # Print statistics
    print(f"Correct matches: {correct}")
    print(f"Total lines: {total}") 
    print(f"Accuracy: {(correct/total*100):.2f}%")

# Example usage
input_text = """



"""

check_matches(input_text)