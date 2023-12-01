import utils
import chatbot

if __name__=='__main__':
    argv = utils.parse()
    if argv.train:
        if argv.train_file is None:
            print('When training mode, training chatlog is required!')
            exit(0)
        chatbot.train(argv)
    elif argv.run:
        chatbot.run(argv)
    else:
        print('Specify mode --run or --train')
    exit(0)
