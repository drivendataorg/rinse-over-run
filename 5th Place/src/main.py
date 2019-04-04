
from src.features.utils import get_datasets
import copy
from src.models.model_1l1 import train_model_1
from src.models.model_2l1 import train_model_2
from src.models.model_3l1 import train_model_3
from src.models.model_l2 import train_l2
from src.config import DEBUG_RUN

def main():
    datasets,labels=get_datasets(debug=DEBUG_RUN)
    train_model_1(copy.deepcopy(datasets), labels)
    train_model_2(copy.deepcopy(datasets), labels)
    datasets2=train_model_3(copy.deepcopy(datasets),labels)
    train_l2(datasets2)

if __name__=='__main__':
    main()