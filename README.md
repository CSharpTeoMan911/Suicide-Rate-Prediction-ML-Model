# Suicide rate prediction ML model using Multivariate Linear Regression

## Prerequisites

* Conda
* Jupyter Notebook


### Dependencies

![Dependencies](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/580c65b4-de50-44e5-be8e-9e4581af9f8a)

<br>

* Pandas is required for processing **.csv** files into **DataFrame** objects.
* Numpy is required for processing numrerical data, transforming and manipilating numerical data, and initiating functions regarding numerical data oeprations.
* Seaborn is required for visualising datasets and the relationships between data categories
* Sklearn is required to process ***Linear Regression**, evaluate the accuracy of the models, and to normalise the data.

### Dataset

![Dataset](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/b0cf1ccc-a6e3-4b72-b333-5f48e3640a36)

<br>

The dataset used was sourced from **Kaggle**. The creator of this dataset used data provided by trusted sources.
The dataset can be downloaded at this [link](https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016). Alternatively, you can download the sourcecode and the dataset in the repository's **Releases** section. 

<br>

![Eva_Capture2020009149](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/04be33b0-7d07-4ead-8921-e8ba3a2cabd1)
![Eva_Capture1124962871](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/bbc59ea2-100a-403b-9f49-2b57f9b61e42)

<br>
<br>
<br>

## Rational

The purpose of this machine learning model is to predict the suicide rate per 100k people of one or more individuals based on their **age** and **gender** in relation with their country of origin. The processes with which the data was manipulated and the process with which the machine learning model was trained are explained in detail below.

### Data Wrangling

![Dataset with problematic value types](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/8235075d-abfb-45d1-ae70-5b63a78ba1c1)

<br>

The dataset has multiple problematic data types from the from the perspective of a **Multivariate Linear Regression Model**. The aforementioned fact is caused by the fact that in the **Multivariate Linear Regression**, the formula used to calculate the predicted **Y** values based on the provided **X** values is:&nbsp; **&#x0176; = &#x2205;&#x2080; + &#x2205;&#x2081;X&#x2081; + &#x2205;&#x2081;X&#x2082; . . . + &#x2205;&#x2093;X&#x2093;**. As a result, the **X** values must be numeric values in order to calculate the previously shown formula and the **&#x0176;** values themselves, and as it can be seen in the picture that shows the values in the dataset, the **sex** and **age** columns have non-numerical values. This values are critical to this machine learning model, because these values will be used to predict the suicide rate for people based on their sex and age in relation with their country they are living into. In order to be able to calculate the **&#x2205;** values and the **&#x0176;** values, we must represent these **X** values as numerical values.

![Data Wrangling](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/06c443a1-441e-4e08-a545-bbbb672b5f6c)

<br>

In order to represent these **X** values as numerical values, we must analyse the data within the **sex** and **age** columns. The aforementioned columns contain values that are classifications. The **sex** column contains values related to the gender and these values are **male** and **female**, and the **age** column contains values related to age groups. 

![Age classification](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/2803e0c6-704c-4b72-ba1a-3e4aaf6e4338)


To transform the age range classifications into numerical values that retain their meaning and also allow the **&#x2205;** values to be calculated, the smallest number of the age range will be assigned in a different column for each individual row, for example, if a row has the **age** value the age range **15-24 years**, the first two characters of the age range will be selected to represent the classification as a numerical value **( e.g. 15-24 years --> 15 )**. To be sure that no dashes remain after the character selection, due to the fact that some age ranges have as the first number a single digit number we remove the dashes from the two selected characters **( e.g. 5-14 --> 5- --> 5 )**.  

![Sex classification](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/c9d526a3-a24c-4c02-b13d-2d83fdbb9e11)

To transform the sex classifications into numerical values that retain their meaning and also allow the ∅ values to be calculated, the sex values **male** and **female** will be replaced with **1** and **2** respectively and they will be assigned in a different column for each individual row **( e.g. male --> 1 & female --> 2 )**.

![Data cleaning](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/68bcc433-9b8c-4b0f-aef2-a825099f6147)

After all the non-numeric classifications were replaced with numerical equivalents, the columns that contain data that is not relevant to the machine learning model are removed.  

![Normalise data](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/c542e17d-1e95-4b37-b612-9c05790a797c)

Afterwards the data in the column that contains values about the suicide rate is normalised to remove the bias in the data.

<br>
<br>

### Data Visualisation

![Data visualisation](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/fe2047b4-070f-40ef-9f62-1183906aac8a)

The dataset that was cleaned and manipulated previously is visualised. Due to the fact that each country has its own socio-economic factors, the machine learning model will be trained to detect the suicide rate based on gender and age only in relation with the location of the individuals in order to maximise the model's accuracy. To see how fit a **Multivariate Linear Regression** algorithm is for this task, all the sex and age values that are within UK were took to be plotted against the suicide rate  using scatterplot. This was done to visualise the spread of the data, if the data its condenssed, the **Multivariate Linear Regression** accuracy is lowered, otherwise the **Multivariate Linear Regression** accuracy is increased. As it can be seen, the data is distributed pretty well, so the **Multivariate Linear Regression** algorithm will have a decent degree of accuracy.

<br>
<br>

### Implementation of the Multivariate Linear Regression

![Machine learning training](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/30b9b287-3cb4-4a96-88ba-1bf8b0e3b771)

Because the machine learning model's puprose is to predict the suicide rate of individuals using the gender and age values in relation with the country where they are located, all the unique country values from the dataset are extracted. 

![Machine Learning Init](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/00291480-5395-4f70-b77f-f05fa22ff8ca)

<br>

Afterwards, the dictionary that will hold the values regarding the **&#x2205;&#x2080;**, **&#x2205;&#x2081;**, **&#x2205;&#x2082;**, **&#x2205;&#x2083;**, and **r&#x2082;** values, is created.

![Dictionary creation](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/78ef496a-274a-4168-9110-c795745e88a8)

<br>

In order to train the model in accordance with the data of each country, a loop will iterate, train the model with data of each country, and save the values in the aformentioned dictionary.  

![Country data iteration](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/a35add9c-4d30-4e01-81b6-2abaf60bda3f)

After the trainig data is created, the data is cleaned and manipulated. The rows that have an **r&#x2082;** value bellow 50%, and the rows that have **&#x2205;&#x2081;**, **&#x2205;&#x2082;**, and **&#x2205;&#x2083;** values equal to 0 are removed. The removal of rows that have **&#x2205;&#x2081;**, **&#x2205;&#x2082;**, and **&#x2205;&#x2083;** values equal to 0 is done to prevent bias in the data, becuase no country in the world has a suicide rate equal to 0. The removal of rows that have an **r&#x2082;** value bellow 50% is done to remove rows that do not allow the model to have an accuracy greater than 50%.

![Training data wrangling](https://github.com/CSharpTeoMan911/Suicide-Rate-Prediction-ML-Model/assets/87245086/4b121823-d832-4562-bf29-7bb902e1c1d8)

<br>
<br>
<br>
<br>
