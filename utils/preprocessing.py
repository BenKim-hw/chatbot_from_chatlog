import os
import csv
from cleantext import clean

remove_strings = ['https','Photo','Emoticon','샵검색','This message was deleted','Video',
                  'File','Contacts','카카오맵','photos','네이버','#']

def check_corrupt_str(s):
    for i in remove_strings:
        if i in s:
            return True
    return False


def preprocess(argv):
    processed_chat_dir = 'chatlogs_processed/'+argv.train_file+'_processed.csv'
    if os.path.exists(processed_chat_dir):
        print('Preprocessing already complete')
        return
    
    filtered_chat = open(processed_chat_dir,'w')
    filtered_chat.write('Q,A,label\n')
    cnt1 = 0
    tmp1 = ''

    with open('chatlogs_raw/'+argv.train_file+'.csv') as f:
        reader = csv.reader(f)
        next(reader)
        next(reader) # ignore 1st line "Date,User,Message"
        for row in reader:
            for (i,v) in enumerate(row):
                v = v.strip()
                if check_corrupt_str(v): # filter corrputed strings
                    continue
                for c in ["'",'"','@','.',',',';','?','ㅋ','ㄷ','ㅠ','ㅜ','zz','zzz','^','~']: # remove characters
                    v = v.replace(c, '')
                v = clean(v, 
                    fix_unicode=False,
                    to_ascii=False,
                    lower=True,
                    normalize_whitespace=False,
                    no_line_breaks=False,
                    strip_lines=True,
                    keep_two_line_breaks=False,
                    no_urls=False,
                    no_emails=False,
                    no_phone_numbers=False,
                    no_numbers=False,
                    no_digits=False,
                    no_currency_symbols=False,
                    no_punct=True,
                    no_emoji=True)

                if len(v) > 30 or len(v)<5: # remove too short or too long chats
                    continue
                if '\n' in v: # remove strings with linebreaks
                    continue

                tmp1 = tmp1 + v + ','
                if cnt1==1:
                    filtered_chat.write(tmp1+'0\n')
                    tmp1 = ''
                    cnt1-=1
                else: cnt1+=1
    
    filtered_chat.close()
    print('Preprocessing complete, saved file under chatlogs_processed/')
    return



def preprocess_double(argv):
    processed_chat_dir = 'chatlogs_processed/'+argv.train_file+'_processed_double.csv'
    if os.path.exists(processed_chat_dir):
        print('Preprocessing already complete')
        return
    
    filtered_chat = open(processed_chat_dir,'w')
    filtered_chat.write('Q,A,label\n')
    cnt1 = 0
    cnt2 = -1
    tmp1 = ''
    tmp2 = ''

    with open('chatlogs_raw/'+argv.train_file+'.csv') as f:
        reader = csv.reader(f)
        next(reader)
        next(reader) # ignore 1st line "Date,User,Message"
        for row in reader:
            for (i,v) in enumerate(row):
                if i != 2:
                    continue
                v = v.strip()
                if check_corrupt_str(v): # filter corrputed strings
                    continue
                for c in ["'",'"','@','.',',',';','?','ㅋ','ㄷ','ㅠ','ㅜ','zz','zzz','^','~']: # remove characters
                    v = v.replace(c, '')
                v = clean(v, 
                    fix_unicode=False,
                    to_ascii=False,
                    lower=True,
                    normalize_whitespace=False,
                    no_line_breaks=False,
                    strip_lines=True,
                    keep_two_line_breaks=False,
                    no_urls=False,
                    no_emails=False,
                    no_phone_numbers=False,
                    no_numbers=False,
                    no_digits=False,
                    no_currency_symbols=False,
                    no_punct=True,
                    no_emoji=True)

                if len(v) > 30 or len(v)<5: # remove too short or too long chats
                    continue
            #     if line.count('ㅋ')/(len(line)+1) > 0.5:
            #         continue
            #     if line.count('ㅠ')/(len(line)+1) > 0.5:
            #         continue
                if '\n' in v: # remove strings with linebreaks
                    continue

                tmp1 = tmp1 + v + ','
                tmp2 = tmp2 + v + ','
                if cnt1==1:
                    filtered_chat.write(tmp1+'0\n')
                    tmp1 = ''
                    cnt1-=1
                else: cnt1+=1
                if cnt2==-1: cnt2 +=1; tmp2='';
                elif cnt2==1:
                    filtered_chat.write(tmp2+'0\n')
                    tmp2 = ''
                    cnt2-=1
                else: cnt2+=1
    
    filtered_chat.close()
    print('Preprocessing complete, saved file under chatlogs_processed/')
    return