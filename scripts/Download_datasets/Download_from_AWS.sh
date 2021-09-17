# Include the subject in the command line while executing this script -> ex. sh dl.sh subject_list
for CURSUBJ in $(cat $1)
do
    aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG/${CURSUBJ}/ ${CURSUBJ}/ --no-sign-request --recursive
done

