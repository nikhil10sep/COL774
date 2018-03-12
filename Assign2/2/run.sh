#!/bin/sh

if [ $1 -eq 1 ] 
then
	if [ $2 -eq 1 ]
	then	
		python3 predict_naive_bayes.py $3 models/nb1.model $4
	elif [ $2 -eq 2 ]
	then
		python3.4 process.py $3 $3.stopstem 1
		python3 predict_naive_bayes.py $3.stopstem models/nb2.model $4
	elif [ $2 -eq 3 ]
	then
		python3.4 process.py $3 $3.stop 0
		python3 predict_naive_bayes.py $3.stop models/nb3.model $4
	else
		echo Unknown option 2
	fi
elif [ $1 -eq 2 ]
then
	if [ $2 -eq 1 ]
	then
		python3 predict_svm.py $3 0 models/model_pegasos $4
	elif [ $2 -eq 2 ]
	then
		python3 predict_svm.py $3 1 models/model_linear_smo $4
	elif [ $2 -eq 3 ]
	then
		python3 predict_svm.py $3 2 models/smo_model_c_4 $4
	else
		echo Unknown option 2
	fi
else 
    echo Unknown option 1
fi