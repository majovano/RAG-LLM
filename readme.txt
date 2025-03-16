Please make sure first that you have all the requirements in the requirement.txt file.
Please also make sure that you are running on GPU T4.

You can find all the results in tfidf_values, lsi_values and the field_values 

the lsi_values corresponds to LSI

the tfidf_values correponds to the TF-IDF model with full vocabulary 

the field_values correpsonds to all the combination fields 

You can modify the n_neighbors and the n_components in the main.py file to obtain the other metrics in case you need to verify. In the 3 different documents, they are the configurations.

If you want to change the combine fiels simply uncomment one of the version and comment the other versions

-------------------------------------
|To Run the model simply write:     |
|                                   |
|python main.py (or python3 main.py)| 
------------------------------------|

It will execute and show all the metrics that are in the report. For the Language model part every output by the queries are in the report. You can just changes if needed for question 8 ten queries that we had to design just change the indices and it will work. 

I put it into comment this way input_string = prompt + str(retrieved_ten_documents[9])  # + irrelevant_context# + str(retrieved_document[0])



------------------------------------------------------------------------------------------------

You can find the of the part 2 commented because they have not been executed locally but only on jupyter and copilot. Nevertheless the part 1 of the project has been tested locally and is executable. I have added also the different main notebooks that have been used to answer the questions.




