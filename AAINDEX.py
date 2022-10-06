#!/usr/bin/env python
#_*_coding:utf-8_*_

import sys, os, re, platform
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import checkFasta

def AAINDEX(fastas, **kw):
	if checkFasta.checkFasta(fastas) == False:
		print('Error: for "AAINDEX" encoding, the input fasta sequences should be with equal length. \n\n')
		return 0

	AA = 'ARNDCQEGHILKMFPSTWYV'

	fileAAindex = 'AAindex.txt'
	with open(fileAAindex) as f:
		records = f.readlines()[1:]

	AAindex = []
	AAindexName = []
	selected=['SeqPos.1.NADH010107', 'SeqPos.8.AURR980105', 'SeqPos.24.RACS820102', 'SeqPos.4.PALJ810114', 'SeqPos.44.CHOP780201', 'SeqPos.3.ARGP820102', 'SeqPos.35.TANS770104', 'SeqPos.8.GEOR030107', 'SeqPos.41.RICJ880117', 'SeqPos.25.GEOR030104', 'SeqPos.1.NAKH900110', 'SeqPos.48.RACS770103', 'SeqPos.2.PONP800108', 'SeqPos.8.AURR980114', 'SeqPos.1.YUTK870104', 'SeqPos.43.GEIM800110', 'SeqPos.4.MEIH800102', 'SeqPos.1.AURR980107', 'SeqPos.24.GEOR030105', 'SeqPos.44.RICJ880112', 'SeqPos.14.FAUJ880113', 'SeqPos.50.TANS770104', 'SeqPos.7.ROBB760102', 'SeqPos.1.GARJ730101', 'SeqPos.4.CHAM830101', 'SeqPos.44.CHAM830104', 'SeqPos.8.CIDH920103', 'SeqPos.41.TANS770102', 'SeqPos.1.WERD780103', 'SeqPos.1.FINA910101', 'SeqPos.8.FINA770101', 'SeqPos.43.OOBM850102', 'SeqPos.1.LEVM780101', 'SeqPos.18.BUNA790103', 'SeqPos.50.YUTK870104', 'SeqPos.1.PTIO830102', 'SeqPos.4.FAUJ880110', 'SeqPos.8.RICJ880111', 'SeqPos.41.PONP930101', 'SeqPos.2.RICJ880111', 'SeqPos.1.GEOR030107', 'SeqPos.3.GUYH850101', 'SeqPos.44.GEIM800103', 'SeqPos.16.OOBM850102', 'SeqPos.1.BLAS910101', 'SeqPos.48.HOPA770101', 'SeqPos.4.GEIM800108', 'SeqPos.1.YUTK870101', 'SeqPos.44.AURR980114', 'SeqPos.14.JOND750102']
	selected = [x.split('.')[-1].strip() for x in selected]
	for i in records:
		if i.rstrip().split()[0].strip() in selected:
			AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
			AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
	index = {}
	for i in range(len(AA)):
		index[AA[i]] = i

	encodings = []
	header = []
	for pos in range(1, len(fastas[0][1]) + 1):
		for idName in AAindexName:
			header.append('SeqPos.' + str(pos) + '.' + idName)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], i[1]
		code = [name]
		for aa in sequence:
			if aa == '-':
				for j in AAindex:
					code.append(0)
				continue
			for j in AAindex:
				code.append(j[index[aa]])
		encodings.append(code)
	return encodings