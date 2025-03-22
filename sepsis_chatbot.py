import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

class SepsisDetectionChatbot:
    def __init__(self, model_path='sepsis_nn_model.h5', scaler_path='sepsis_prediction_model.pkl'):
        """Initialize the sepsis detection chatbot with your saved model files.
        
        Args:
            model_path: Path to your saved neural network model
            scaler_path: Path to your saved StandardScaler object
        """
        # Load your trained model
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        
        # Load your saved scaler
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Define the features required for sepsis detection based on your dataset
        self.required_features = [
            'Hour', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 
            'BaseExcess', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 
            'Unit2', 'HospAdmTime', 'ICULOS'
        ]
        
        # Initialize patient data dictionary
        self.patient_data = {feature: None for feature in self.required_features}
        self.patient_id = None
        
        # Define normal ranges for each feature for validation
        self.normal_ranges = {
            'Hour': (0, 24),
            'HR': (60, 100),
            'O2Sat': (95, 100),
            'Temp': (36.5, 37.5),
            'SBP': (90, 140),
            'MAP': (70, 100),
            'DBP': (60, 90),
            'Resp': (12, 20),
            'EtCO2': (35, 45),
            'BaseExcess': (-2, 2),
            'Fibrinogen': (200, 400),
            'Platelets': (150, 450),
            'Age': (0, 120),
            'Gender': (0, 1),
            'Unit1': (0, 1),
            'Unit2': (0, 1),
            'HospAdmTime': (-10, 100),
            'ICULOS': (0, 50)
        }
        
        # Current conversation state
        self.current_feature_idx = 0
        self.conversation_state = "greeting"
        
    def get_next_prompt(self):
        """Return the next prompt based on the current conversation state."""
        if self.conversation_state == "greeting":
            self.conversation_state = "patient_id"
            return ("Welcome to the Sepsis Detection Assistant. I'll help you assess a patient's risk of sepsis "
                   "based on vital signs and lab results. Let's start with some information.")
        
        elif self.conversation_state == "patient_id":
            self.conversation_state = "collecting_data"
            return "Please enter the Patient ID for reference:"
        
        elif self.conversation_state == "collecting_data":
            if self.current_feature_idx < len(self.required_features):
                feature = self.required_features[self.current_feature_idx]
                feature_name = feature
                
                # Format the prompt with normal range information
                low, high = self.normal_ranges[feature]
                
                # Special instructions for certain features
                if feature == "Gender":
                    return f"Please enter patient's {feature_name} (0 for female, 1 for male):"
                elif feature in ["Unit1", "Unit2"]:
                    return f"Please enter {feature_name} (0 or 1):"
                else:
                    return f"Please enter the patient's {feature_name} (normal range: {low}-{high}):"
            else:
                self.conversation_state = "confirmation"
                return self._format_data_confirmation()
        
        elif self.conversation_state == "confirmation":
            self.conversation_state = "prediction"
            return "Thank you for providing all the information. Would you like me to analyze the risk of sepsis now? (yes/no)"
        
        elif self.conversation_state == "prediction":
            self.conversation_state = "done"
            return self._get_prediction()
            
        elif self.conversation_state == "done":
            return "Is there anything else you'd like to know about the sepsis risk assessment?"
    
    def process_response(self, user_input):
        """Process the user's response and update the conversation state."""
        if self.conversation_state == "patient_id":
            self.patient_id = user_input
            return self.get_next_prompt()
            
        elif self.conversation_state == "collecting_data":
            try:
                # Convert to float and validate input
                value = float(user_input)
                feature = self.required_features[self.current_feature_idx]
                
                # Check if the value is within a reasonable range
                low, high = self.normal_ranges[feature]
                if feature not in ["Gender", "Unit1", "Unit2"]:  # Skip range check for binary variables
                    if value < low * 0.5 or value > high * 2:
                        return f"The value seems unusual. Are you sure {value} is correct for {feature}? Please confirm or re-enter."
                
                # Special validation for binary features
                if feature in ["Gender", "Unit1", "Unit2"] and value not in [0, 1]:
                    return f"{feature} must be either 0 or 1. Please enter a valid value."
                
                # Store the value and move to the next feature
                self.patient_data[feature] = value
                self.current_feature_idx += 1
                
            except ValueError:
                return "Please enter a valid number."
                
        elif self.conversation_state == "confirmation":
            if user_input.lower() in ["yes", "y", "correct", "confirm"]:
                self.conversation_state = "prediction"
            else:
                # Reset to collect data again
                self.current_feature_idx = 0
                self.conversation_state = "collecting_data"
                return "Let's collect the information again. Starting with the first measurement."
                
        elif self.conversation_state == "prediction":
            if user_input.lower() in ["yes", "y"]:
                return self._get_prediction()
            else:
                self.conversation_state = "done"
                return "No problem. Let me know if you'd like to analyze the data later."
                
        # Get the next prompt
        return self.get_next_prompt()
    
    def _format_data_confirmation(self):
        """Format the collected data for confirmation."""
        confirmation_msg = f"Please confirm the following information for Patient ID: {self.patient_id}\n\n"
        
        for feature, value in self.patient_data.items():
            confirmation_msg += f"- {feature}: {value}\n"
            
        confirmation_msg += "\nIs this information correct? (yes/no)"
        return confirmation_msg
    
    def _get_prediction(self):
        """Get the sepsis risk prediction from the model."""
        if self.model is None:
            return "Error: No model has been loaded for prediction."
            
        # Check if all required data is available
        if None in self.patient_data.values():
            return "Error: Not all required information has been provided."
            
        # Prepare data for the model
        features = np.array([[self.patient_data[feature] for feature in self.required_features]])
        
        # Add SepsisLabel as a placeholder if your model expects it (will be predicted)
        # This assumes your model was trained with this feature order
        
        # Scale features
        features_scaled = self.scaler.transform(features)
            
        # Make prediction
        prediction = self.model.predict(features_scaled)[0][0]
        
        # Format the response based on the risk level
        risk_level = self._get_risk_level(prediction)
        
        response = f"Sepsis Risk Analysis for Patient ID: {self.patient_id}\n\n"
        response += f"Risk probability: {prediction:.2f} ({risk_level})\n\n"
        
        # Add clinical guidance based on risk level
        if risk_level == "High Risk":
            response += "CLINICAL RECOMMENDATION: Immediate medical attention is needed. "
            response += "Consider initiating sepsis protocol and obtaining blood cultures, lactate measurement, "
            response += "and starting broad-spectrum antibiotics within 1 hour if sepsis is suspected."
        elif risk_level == "Moderate Risk":
            response += "CLINICAL RECOMMENDATION: Close monitoring is advised. "
            response += "Consider additional diagnostic tests, more frequent vital signs monitoring, "
            response += "and re-evaluation within 1-2 hours."
        else:
            response += "CLINICAL RECOMMENDATION: Continue routine monitoring. "
            response += "Re-evaluate if there are changes in patient condition or new symptoms develop."
            
        response += "\n\nIMPORTANT: This is a decision support tool only and should not replace clinical judgment. "
        response += "Always consult with a healthcare professional for proper diagnosis and treatment."
        
        return response
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level."""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"

def run_chatbot():
    """
    Run the sepsis detection chatbot with the saved model and scaler
    """
    try:
        # Initialize the chatbot with your saved model and scaler
        chatbot = SepsisDetectionChatbot('sepsis_nn_model.h5', 'sepsis_prediction_model.pkl')
        
        print("=" * 80)
        print("Sepsis Detection Assistant")
        print("=" * 80)
        
        # Start conversation
        print(chatbot.get_next_prompt())
        
        while True:
            user_input = input("> ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Thank you for using the Sepsis Detection Assistant. Goodbye!")
                break
                
            response = chatbot.process_response(user_input)
            print(response)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("If you continue to experience issues, please check that your model and scaler files are in the correct location.")

if __name__ == "__main__":
    run_chatbot()