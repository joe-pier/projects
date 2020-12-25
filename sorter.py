import os
import shutil

for i in os.listdir('../../Google Drive/Università/Università - Bocconi/SMISTAMENTO/SMISTAMENTO'):

    #words= i.split(' ')
    words = i.split('.')
    keyword = words[0].split(' ')
    for j in keyword:
        if j == 'finanza' or j=='FINANZA':
            shutil.move('../../Google Drive/Università/Università - Bocconi/SMISTAMENTO/SMISTAMENTO/'+i ,
                        '../../Google Drive/Università/Università - Bocconi/20198 - Finanza Aziendale e dei Mercati/VARIE/SMISTAMENTO')
            print('ho smistato ' + i+ ' in '+ 'finanza')
        else:
            if j=='bilancio' or j=='BILANCIO':
                shutil.move('../../Google Drive/Università/Università - Bocconi/SMISTAMENTO/SMISTAMENTO/' + i,
                        '../../Google Drive/Università/Università - Bocconi/20126 - Bilancio e Comunicazione Economica/VARIE/SMISTAMENTO')
                print('ho smistato ' + i + ' in ' + 'bilancio')
            else:
                if j=='corporate' or j=='CORPORATE GOVERNANCE' or j=='GOVERNANCE':
                    shutil.move('../../Google Drive/Università/Università - Bocconi/SMISTAMENTO/SMISTAMENTO/' + i,
                                '../../Google Drive/Università/Università - Bocconi/20123 - Sistemi di Corporate Governance/VARIE/SMISTAMENTO')
                    print('ho smistato ' + i + ' in ' + 'corporate governance')
                else:
                    if j=='analisi dei dati' or j=='ANALISI':
                        shutil.move('../../Google Drive/Università/Università - Bocconi/SMISTAMENTO/SMISTAMENTO/' + i,
                                    '../../Google Drive/Università/Università - Bocconi/20179 - Analisi dei Dati/VARIE/SMISTAMENTO')
                        print('ho smistato ' + i + ' in ' + 'analisi dei dati')
                    else:
                        print('il file '+i+' non ha corrispondenze nel registro')











