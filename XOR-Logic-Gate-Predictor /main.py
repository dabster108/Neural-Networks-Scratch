# main.py
import numpy as np
from neural import forward
# Import the trained model and data from trainxor.py
import trainxor

def make_predictions(params, X, Y):
    """Make predictions using trained network"""
    
    # Final forward pass
    Z1, A1, Z2, A2 = forward(X, params["W1"], params["b1"], params["W2"], params["b2"])
    
    # Get predictions
    preds = (A2 >= 0.5).astype(int)
    
    print("\n" + "=" * 50)
    print("XOR Neural Network - Final Predictions")
    print("=" * 50)
    print(f"{'Input':<15} {'Predicted':<12} {'Actual':<10} {'Confidence'}")
    print("-" * 50)
    
    for i in range(len(X)):
        confidence = A2[i][0] if preds[i][0] == 1 else 1 - A2[i][0]
        print(f"{str(X[i]):<15} {preds[i][0]:<12} {Y[i]:<10} {confidence:.4f}")
    
    # Calculate accuracy
    accuracy = np.mean(preds.flatten() == Y) * 100
    print("-" * 50)
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)
    
    return preds

def test_custom_input(params):
    """Test network with custom inputs"""
    print("\n" + "=" * 50)
    print("Test Custom Inputs")
    print("=" * 50)
    
    while True:
        try:
            inp = input("\nEnter two binary values (0 or 1) separated by space (or 'q' to quit): ")
            if inp.lower() == 'q':
                break
            
            values = list(map(float, inp.split()))
            if len(values) != 2:
                print("Please enter exactly 2 values!")
                continue
            
            X_test = np.array([values])
            _, _, _, A2 = forward(X_test, params["W1"], params["b1"], params["W2"], params["b2"])
            prediction = (A2 >= 0.5).astype(int)[0][0]
            confidence = A2[0][0] if prediction == 1 else 1 - A2[0][0]
            
            print(f"Input: {values}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")
            
        except ValueError:
            print("Invalid input! Please enter numbers only.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    # Use the trained parameters from trainxor.py
    print("\n" + "=" * 50)
    print("Using trained model from trainxor.py")
    print("=" * 50)
    
    # Get trained parameters and data
    params = trainxor.params
    X = trainxor.X
    Y = trainxor.Y
    
    # Make predictions on training data
    predictions = make_predictions(params, X, Y)
    
    # Optional: Test with custom inputs
    test_choice = input("\nWould you like to test with custom inputs? (y/n): ")
    if test_choice.lower() == 'y':
        test_custom_input(params)
    
    print("\nProgram completed successfully!")
