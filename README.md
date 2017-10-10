# Human Activity Recognition

The increase in living standards in our society has brought some mixed blessings. On the one hand, we have abundant food supplies (in some parts of the world), and we tend to live longer, but on the other hand, our sedentary lifestyle has lead to an obesity epidemic and seems linked to sharp decreases in quality of life during our golden years.

With the advent of wearable devices (i.e., Fitbit), the field of human activity recognition (HAR) has gotten a boost. Our hope is that accurate monitoring of human activity will lead us to a better understanding of the link between a sedentary lifestyle and its effects on our health, and also we hope that it should allow us to gently nudge individuals towards a more active lifestyle using by monitoring and positive reinforcement.

Activity identification presents many challenges, one being how to account for the differences in body sizes which leads to individuals displaying different ranges of motion when performing any one activity.

In this project my goal was to determine if it is feasible that wearable devices can be used to determine what activities you are doing.

Please feel free to check out the corresponding blog post for more information on the process I followed.
https://bernardomesa.github.io/DoesFitbitWork/

# Conclusion
It is possible to reach 94% accuracies in identifying 12 different activities when using a setup of 3 separate wearable devices worn simultaneously on the ankle, chest, and wrist.

This accuracy dropped to the range 88% to 91% when using a combination of only 2 of these devices, and further dropped to the low 80% when using only 1 of these devices.

# Further potential work
I would like to give boosting another try. After working on this project a colleague mentioned he usually sets the number of states to the 1000's in a boosting ensemble and allows for early stopping when the accuracy has not improved for something like 100 iterations, and he does not worry to much about the other parameters in the ensemble.


## The Dataset

I set out to test if we can use the sensor signals in wearable devices to determine what activity we are performing. To do this, I used a dataset that comprised the data from 18 different activities, performed by nine individuals wearing three inertial devices.

One of the inertial devices was worn on the ankle, another one on the wrist, and another around the chest area.


## Download PAMAP2_Dataset

On the command line, navigate to the folder where you have saved the HumanActivityRecognition.ipynb file 
Run the following commands on your command line:

- wget http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip
- sudo apt-get install unzip
- mkdir destination_directory
- unzip PAMAP2_Dataset.zip -d <destination_directory>


# File description

1. HumanActivityRecognition.ipynb - Main file that details data preparation, exploration, and modeling.
2. lightGBM.ipynb -  grid search using lightGBM algorithm. Same as in section 'GRID SEARCH - LightGBM' in file 1.
3. randomizedSearchCV.py - contains python script to run grid search over dataset using 'random forest' and 'gradient boosting classifier' estimators. I used this script to run search on AWS server on a 40 core instance.


