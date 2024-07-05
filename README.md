# **deep-learning-challenge**


## *1) Overview of the Analysis*
This analysis aims to develop a machine learning neural network model, specifically a binary classifier, to assist Alphabet Soup in selecting grant recipients with the highest probability of success. The model will leverage historical data on past funding recipients to identify key factors associated with successful ventures. 


## *2) Results*

* ### **Data Preprocessing:**

    * **Target Variable:** 

        The target variable that was used for this neural network model was the "IS_SUCCESSFUL" column (0 => WAS NOT successful | 1 => WAS successful). Since the goal of the analysis was to build a model that is able to accurately predict applicants that had the best chance of being success in their ventures, "IS_SUCCESSFUL" was the perfect target variable for achieving this goal. 

    * **Feature Variables:** 

        The features that were used in the original model were "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", and "ASK_AMT". These features gave diverse information that the model could find complex relationships between to help build the weights between each node to give accurate predictions. "NAME" was eventually used in the optimized model but that will be discussed later in the results section. 

    * **Irrelevant Data:**

        The final two columns of data that were included in the dataset that were not used as features for the original model was the "EIN" and "NAME" column. The "EIN" column is a unique identifier for each row (AKA - each applicant in the dataset) but although the "NAME" column is not necessarily unique for each record becuase some applicant names can have multiple applications for their different ventures, this column was still left out as a feature for the first original model that was created. 

* ### **Compiling, Training, and Evaluating the Model:**

    * **Neurons, Layers, and Activation Functions Used for Neural Network Model:**
        
        * **Neurons:**

            The original model utilized 16 neurons in each hidden layer with a single neuron in the output layer. This configuration achieved the highest accuracy with the lowest loss function value after exploring various architectures with a number of neurons equal to powers of 2 (i.e., 2, 4, 8, 16, etc.).  This approach maximizes computational efficiency.

        * **Layers:**

            The original model architecture employs two hidden layers and a single output layer. This configuration was chosen after considering various architectures. Two hidden layers provide a balance between model complexity and capacity. A single hidden layer might struggle to capture the intricacies of the data, while a significantly deeper architecture could lead to overfitting.

        * **Activation Functions:**

            The original model employs ReLU (Rectified Linear Unit) activation functions in the two hidden layers and a sigmoid activation function in the output layer. ReLU offers advantages like computational efficiency and the ability to address the vanishing gradient problem that can hinder training in deeper networks. Since this model is a binary classification task aiming to predict successful (1) or unsuccessful (0) outcomes, a sigmoid activation function is appropriate due to sigmoid outputs values between 0 and 1, which can be interpreted as probabilities of an application being successful.

    * **Target Model Performance / Was it Possible?:**

        The target model performance was 75% accuracy before the model was built. The original model was 72.6% which did not meet the standards that Alphabet Soup was expecting. This meant that the model had to be optimized to try and reach the goal of atleast 75% accuracy before the model can be put to use. 


    * **Steps Taken to Increase Model Performance / Optimization of Model:**

        * **Step 1:** The first step that was taken to increase the performance of the model was to use "NAME" as a feature instead of leaving that data out of the model. "NAME" values that had a count of less then 49 were put into a seperate "other" bin to avoid having two many categorical features and have the model learn from more common occurrences. 


        * **Step 2:** The second step that was taken was to increase the number of categorical features for the "APPLICATION_TYPE" column by changing the cutoff value less than 16 and have the rest of the data in the "other" bin. As well as increasing the number of categorical features for the "CLASSIFICATION" column by changing the cutoff value to less than 100. 


        * **Step 3:** The third step that was taken was to add two more hidden layers to the model (four total hidden layers) and experimenting with the neurons for each layer to see which combination gave the best performance. (First Hidden Layer => 16 neurons, Second Hidden Layer => 32 neurons, Third Hidden Layer => 16 neurons, Fourth Hidden Layer => 32 neurons). 


        * **Step 4:** The final step that was taken was to change the activation function for the first two hidden layers to "tanh", the last two hidden layers as "relu", and keep "sigmoid" for the final output layer After experimenting with the different activation functions, this combination gave the best results and helped increase the performance of the model. 






## *3) Summary*

* ### **Overall Results:**
The model did a good job of predicting Alphabet Soup applicants with the best chances of success in their ventures. After optimizing the model, the accuracy score improved to 76.23% which was above the minimum threshold of 75% that Alphabet Soup was aiming for. The model's loss function employed binary cross-entropy, which inherently favors outcomes closer to 0. The calculated loss of 0.4901 represents the average error between the model's predictions and the actual classifications (successful or unsuccessful funding). Overall, a suitable model was created for Alphabet Soup and although the model meets their requirments, it can still be improved. 



* ### **Different Model Reccomendation:**
Given the complexity of the data, a neural network model appears to be a well-suited choice for Alphabet Soup's use case. Applicant success after receiving funding likely hinges on a multitude of factors that may not be entirely captured in the current dataset. The features and information used to train the model could exhibit intricate relationships that a neural network is adept at recognizing compared to simpler classification models like Logistic Regression, Random Forests, or K-Nearest Neighbors. While there's always room for improvement in the current model's accuracy, for Alphabet Soup's specific use case, the inherent complexity of the data and the potential for non-linear relationships between features make neural networks a compelling option for identifying the most relevant factors associated with successful ventures.