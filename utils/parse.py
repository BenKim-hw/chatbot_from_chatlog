import argparse

def parse():
    parser = argparse.ArgumentParser(description='CHATBOT_FROM_CHATLOG')
    parser.add_argument('-r','--run', action='store_true', default=False)
    parser.add_argument('-rm', '--run_model')

    parser.add_argument('-t','--train', action='store_true', default=False)
    parser.add_argument('-tf', '--train_file')
    parser.add_argument('-te', '--train_epochs', type=int, default=15)
    
    argv = parser.parse_args()

    return argv