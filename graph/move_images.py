import os, random, shutil
#source ='..\\data_4graphs\\train\\short'
#destination ='..\\data_4graphs\\test\\short'
def move(source, target):
    files = [filenames for (filenames) in os.listdir(source)]
    random_file = random.choice(files)
    shutil.move(f'{source}\\{random_file}', target)
if __name__ == '__main__':
    train_long = '..\\data\\train\\long'
    validation_long = '..\\data\\validation\\long'
    test_long = '..\\data\\test\\long'

    train_short ='..\\data\\train\\short'
    validation_short ='..\\data\\validation\\short'
    test_short ='..\\data\\test\\short'

    train_n = '..\\data\\train\\neutral'
    validation_n = '..\\data\\validation\\neutral'
    test_n = '..\\data\\test\\neutral'

    temp = '..\\temp'

    for x in range(14423):
        #move(train_long,test_long)
        #move(train_short,test_short)
        #move(train_n,test_n)
        #move(train_long,validation_long)
        #move(train_short,validation_short)
        #move(train_n,validation_n)

        move(train_n,temp)