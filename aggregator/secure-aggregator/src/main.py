# This file will be the entry point of secure aggregation container

# The container will periodically read the model updates from its local volume.
#   It will read the data from a shared volume with reporter container (if the trainers are from local cluster)
#   It will receive the from other nodes and then store it locally before secure aggregation 
