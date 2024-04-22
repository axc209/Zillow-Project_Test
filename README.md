This project incorporates using Rapid API to get housing data. You'll need to subscribe or pay to pull over 10,000+ requests, free version is like a 100. So if you're using the free version I suggest inserting one location or pull 100 random "zpid" from different locations.

In the python script you'll need to change the locations you desire and also insert your own API key. The housing data information will be pulled per page, then history by "zpid", and then housing details by "zpid".
Also, it will save an excel output to your directory.

As for the regression script this will incorporate different methods of predictive modeling such as linear/lasso regression, random forest, and XGBoost. It will cover, optimizazation of these models, pruning, cross validation, and go over if the model is
overfitting. So that we can have the best model available for this project. However, you're more than welcome to pull in more data from other sources to increase the predictability of your model.

The PowerBI file has my dashboard (however you can change it to however you like). It does has DAX syntax of creating a new table, and building different types of metrics just to display knowledge in DAX. 

NOTE: 
1.) The IDE I'm using is Spyder, so you may need to tweak the code a little bit for some IDEs
2.) I'll also have another script, in the future, doing the same thing but doing an ETL process/utilizing Azure
3.) Will also create a public Tableau Dashboard replicating my PowerBI Dashboard and also display the use of the Tableau Syntax
