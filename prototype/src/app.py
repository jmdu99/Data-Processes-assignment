import joblib
import os
import numpy as np
import sys
from warnings import simplefilter

simplefilter("ignore")

src_path = os.path.dirname(os.path.realpath(__file__))
root_path = src_path.replace("src","")
resources_path = root_path + "resources"
    
def load_data():
    # Load trained scaler (scaler.pkl) in survival prediction 
    # (data_processes_assignment.py) in order to scale user input data 
    # in the same way as training data 
    scaler = joblib.load(resources_path + '/scaler.pkl')
        
    # Load selected Random Forest trained model (random_forest.pkl) in survival prediction 
    # (data_processes_assignment.py) in order to make new predictions
    model = joblib.load(resources_path + '/random_forest.pkl')

    '''
    Selected model: Random Forest
    Selected features for this model: ['DAYS_HOSPITAL', 'DAYS_ICU', 'SAT_O2']
    Mean accuracy: 0.853 (std deviation: 0.021)
    Total confusion matrix:
    [[1309   57]
     [ 183   82]]
    '''
    return scaler,model

def init_app(scaler,model):
    print("\nWelcome to the 'COVID-19 survival prediction' prototype")
    while True:
        answer = input("Do you want to know if a patient suffering covid will survive? (Enter 'yes' or 'no'): ")
        if answer == "yes": 
            print("\nIn that case, it is necessary to know the following patient data:")
            age = input("Age of the patient (in years): ")
            sex = input("Sex of the patient (enter '0' if female or '1' if male): ") 
            days_hospital = input("Number of days the patient has been in the hospital (enter '0' if the patient has not been there any day): ")  
            days_icu = input("Number of days the patient has been in ICU (enter '0' if the patient has not been there any day): ")  
            temp = input("Patient temperature (in Celsius degrees): ")  
            heart_rate = input("Patient heart rate (number of beats per minute): ")  
            sat_O2 = input("Patient blood oxygen level (percentage, from '0' to '100'): ")
            blood_pres_sys = input("Patient systolic blood pressure (in mm Hg): ")
            blood_pres_dias = input("Patient diastolic blood pressure (in mm Hg): ")
            
            # User input in a nested list
            user_input = [[age,sex,days_hospital,days_icu,temp,heart_rate,sat_O2,blood_pres_sys,blood_pres_dias]]
            
            '''
            Format:
                [[AGE,SEX,DAYS_HOSPITAL,DAYS_ICU,TEMP,HEART_RATE,SAT_O2,BLOOD_PRES_SYS,BLOOD_PRES_DIAS]]
                
            SEX: FEMALE -> 0 ; MALE -> 1 
            EXITUS: NO -> 0 ; YES -> 1   
            
            Input example: 
                [[15,0,4,0,37,88,92,123,71]]
                
            Output after scaling:
                array([[0.        , 0.        , 0.07843137, 0.        , 0.57575758,
                        0.36434109, 0.88135593, 0.47945205, 0.41666667]])
                
            Model input: [[0.07843137, 0.,        0.88135593]]
            Model output: array([0]) -> Will survive
            '''
            
            # Scale it
            scaler.clip = False
            scaled_input = scaler.transform(user_input)
            
            # Delete unnecessary featues for the model
            model_input = np.delete(scaled_input, [0,1,4,5,7,8], 1)
            
            # Model predicion
            exitus = model.predict(model_input)
            print("\nDIAGNOSIS:")
            if exitus == [0]:
                print('Based on the trained model, the patient is most likely to survive the covid.')
            else:
                print('Based on the trained model, the patient is most likely to NOT survive the covid.')
            print('\nWARNING: this diagnosis has been made by a statistical model and therefore there is a possibility that it may be wrong. Take only as a reference.')
            while True:    
                end = input("\nExit the app? (Enter 'yes' or 'no'): ")
                if end == "yes":
                    print("See you soon!")
                    sys.exit()
                elif end == "no": 
                    break;
                else: 
                    print("Please enter 'yes' or 'no'.")
        elif answer == "no": 
            print("See you soon!")
            sys.exit()
        else: 
            print("Please enter 'yes' or 'no'.") 
        
if __name__ == '__main__':
    scaler,model = load_data()
    init_app(scaler,model)
