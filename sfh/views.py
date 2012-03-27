from django.shortcuts import render_to_response
from sfh.models import Train, Tide
from math import fabs, sqrt, pow, sin, radians, pi
from random import randint
import matplotlib.pyplot as plt
from itertools import combinations

#
# To-Do on thursday(23.02):
#    - proper knn with different k's (done :) )
#    - check accuracy - cross-validation (done :) )
#######################################################
# To-Do on Friday(24.02):
#    - knn - add tide level to distance measuring
#    - find accurate tide level information
########################################################
# To-Do on Monday(27.02):
#    - Meeting with Mahesh - 4.30pm, ask: Tide level 
#      training
#


#########################################################################
# 									#
# index():								#
#  - send the data that needs to be classified in GET request:		#
#     - tch - transmitting channel 					#
#     - thr - throughput of tch						#
#									#
#########################################################################

def index(request):
    if request.method == 'GET':
        data=request.GET
        if 'tch' in data:
            tch = data.get('tch')
        if "thr" in data:
            thr = float(data.get('thr'))
        result = knn(tch, Train.objects.values(), { 'throughput' : thr })           
        return render_to_response('sfh/index.html', { 'knn' : result })
    return render_to_response('sfh/index.html')

#################################################################################################
#												#
# knn(thr, trains, k):										#
#  - perform k-nearest neighbor algorithm							#
#  - argumets:											#
#     - tch    - transmitting channel								#
#     - trains - a traing set									#
#     - attributes - dictionry holding values to calculate the dist				#
#     - k      - number of nearest neigbors looking for, by default 3 				#
#												#
#################################################################################################

def knn(tch, train_s, attributes=None, k=3):
    neighbors = [ dict(zip(attributes.keys(), [100000 for x in range(len(attributes))])) for i in range(k)]
    ch_train = [ t for t in train_s if t.get('transmitting_channel') == float(tch) ]
    c_x=attributes.values()
    for t in ch_train:
        index = k
        for i in range(k):
            n_x=[neighbors[k-1-i].get(x) for x in attributes.keys()] #change a dict to a vector/list
            t_x=[t.get(x) for x in attributes.keys() ]
            if dist(t_x, c_x) < dist(n_x, c_x):
                index = k-1-i
                continue
            else:
                break
        if index < 3:
            neighbors.insert(index, dict([ [ key,t.get(key) ] for key in attributes.keys() ] +
                                     [ [ 'opt_ch_t_thr', t.get('opt_ch_t_thr') ]       ] ))
            neighbors.pop()
    classes = [ n.get('opt_ch_t_thr') for n in neighbors ]
    counters = [ classes.count(classes[i]) for i in range(k) ]
#    print neighbors
    return classes[counters.index(max(counters))]

#################################################################################################
#												#
#  dist(x1,x2):											#
#   - x1, x2 - lists of coordinates in N dimension space where N is len	of x1 and x2		#
#   - returns the square root of the sums of the squares of diffrences between coordinates -	#
#       distance from x1 to x2									#
#												#
#################################################################################################

def dist(x1, x2):
    return sqrt(sum( [ pow(x2[i]-x1[i],2) for i in range(len(x1)) ] ))
#########################################################
#							#
# data():						#
#  - get data from get request and put it to database:	#
#     - ts - timestamp					#
#     - tch - transmitting channel			#
#     - thr - throughput				#
#     - ssnr - selfSNR					#
#     - snoise - self noise				#
#     - osnr - other snr				#
#     - onoise - other noise				#
# 							#
#########################################################

def data(request):
    if request.method == 'GET':
        data = request.GET
        logic = 'ts' in data and 'tch' in data and "thr" in data and 'ssnr' in data 
        logic = logic and 'snoise' in data and 'osnr' in data and 'onoise' in data
        if logic:
            timestamp = (data.get('ts'))
            transmitting_channel = (data.get('tch'))
            throughput = (data.get('thr'))
            self_snr = float(data.get('ssnr'))/float(100)
            self_noise = float(data.get('snoise'))/float(100)
            self_rssi = self_snr+self_noise
            other_snr = float(data.get('osnr'))/float(100)
            other_noise = float(data.get('onoise'))/float(100)
            other_rssi = other_snr+other_noise
            line = Train(timestamp=timestamp, 
                         transmitting_channel=transmitting_channel,
                         throughput=throughput,
                         self_snr=self_snr,
                         self_noise=self_noise,
                         self_rssi=self_rssi,
                         other_snr=other_snr,
                         other_noise=other_noise,
                         other_rssi=other_rssi)
            line.save()
            return render_to_response('sfh/data.html', {'data' : line})
    return render_to_response('sfh/data.html')

#################################################################################
#										#
# tide(request):								#
#  - receiveing the tide information from GET request:				#
#      a) ts - timestamp							#
#      b) tide - tide height							#
#  - To-Do: download it straight from the server				#
#										#
#################################################################################

def tide(request):
   if request.method == 'GET':
        data=request.GET
        if 'ts' in data and 'tide' in data:
            ts = int(data.get('ts'))
            h = float(data.get('tide'))/100
            line = Tide(timestamp=ts, height=h)
            line.save()
            return render_to_response('sfh/tide.html', { 'tide' : line })
   return render_to_response('sfh/tide.html', {'tide' : Tide.objects.all() })

#########################################################################################
#											#
# mix(request):										#
#  - assuming that tide behaviour is linear, computing the height of tide between two 	#
#    entries in Tide db									#
#  - To-Do: use sin interpolation(done),						#
#  - Can add GET variables to use diff approximation					#
#											#
#											#
#  A*sin(B*(x - C)) + D = y 								#
#											#
#											#
#											#
#########################################################################################

def mix(request):
    print "in mix"
    all_tides = Tide.objects.all()
    all_trains = Train.objects.all()
    counter = 0
    for tr in all_trains:
        prev_td = Tide(0,0)
        for td in all_tides:
            if tr.timestamp < td.timestamp:
                A = (td.height - prev_td.height)/2
                period = 2*radians(td.timestamp-prev_td.timestamp)
                B = 2*pi/period
                C = radians(td.timestamp) - (period/4)
                D = (td.height+prev_td.height)/2
                ctd = A*sin(B*(radians(tr.timestamp)-C)) + D
                tr.tide_level = ctd
                tr.save()
                break
            prev_td = td
        counter += 1
        if counter % 20 == 0:
            print len(all_trains)-counter, "left"
    return render_to_response('sfh/mix.html', {'trains' : all_trains})

###########################################################################################################
#	Don't use it!!											  #
# dummy(request):											  #
#  - adding dummy entries so total numeber of entries is a multiply of number of channels used		  #
#    but is adding a 0 throughput is the best idea? maybe should add -1 throughput or -infinity throghput #
#  													  #
###########################################################################################################

def dummy(request):
    CHANNELS = [ 5.18, 5.26, 5.32, 5.5, 5.6, 5.7, 5.745, 5.785, 5.825 ]
    all_train = Train.objects.all()
    collect = []
    prev_tr = all_train[0]
    count = 0
    for tr in all_train:
        if not tr.transmitting_channel in collect:
            collect.append(tr.transmitting_channel)
        else:
            for ch in CHANNELS:
                if not ch in collect:
                    t = Train(timestamp=prev_tr.timestamp, transmitting_channel=ch, throughput=0, tide_level=prev_tr.tide_level)
                    t.save()
            collect = [tr.transmitting_channel]
        prev_tr = tr
        if count % 20 == 0:
            print len(all_train)-count, "left"
        count += 1
    if not len(collect) == len(CHANNELS):
        for ch in CHANNELS:
            if not ch in collect:
                t = Train(timestamp=prev_tr.timestamp, transmitting_channel=ch, throughput=0, tide_level=prev_tr.tide_level)
                t.save()

    return render_to_response('sfh/dummy.html', {'dummies' : Train.objects.all() })

#################################################################################################
#												#
# train(request):										#
#  - for every slot check which channel has the best throughput and set all optimal channels	#
#  in this slot to this channel									#
#  - To-Do: Different time slots of max length of len(CHANNELS) (done!)				#
#-----------------------------------------------------------------------------------------------#
#  - To-Do: in get var what kind of train is needed - tide or throughput			#
#  talk with mahesh about tide training								#
#												#
#################################################################################################

def train(request):
    print "in train"
    counter = 0
    thr = -1000
    ssnr = -1000
    osnr = -1000
    srssi = -1000
    orssi = -1000
    ch_t = 0
    ch_ssnr = 0
    ch_osnr = 0
    ch_srssi = 0
    ch_orssi = 0
    slot = []
    t_channels = []
    train_s = Train.objects.all()

    while counter < len(train_s):
        entry = train_s[counter]
        if not entry.transmitting_channel in t_channels:
            t_channels.append(entry.transmitting_channel)
            slot.append(entry)
            if entry.throughput > thr:
                thr = entry.throughput
                ch_t = entry.transmitting_channel
            if entry.self_snr > ssnr:
                ssnr = entry.self_snr
                ch_ssnr = entry.transmitting_channel
            if entry.self_rssi > srssi:
                srssi = entry.self_rssi
                ch_srssi = entry.transmitting_channel
            if entry.other_snr > osnr:
                osnr = entry.other_snr
                ch_osnr = entry.transmitting_channel
            if entry.other_rssi > orssi:
                orssi = entry.other_rssi
                ch_orssi = entry.transmitting_channel
        else:
            for t in slot:
                t.opt_ch_t_thr = ch_t
                t.opt_ch_thr = thr
                t.opt_ch_t_ssnr = ch_ssnr
                t.opt_ch_ssnr = ssnr
                t.opt_ch_t_srssi = ch_srssi
                t.opt_ch_srssi = srssi
                t.opt_ch_t_osnr = ch_osnr
                t.opt_ch_osnr = osnr
                t.opt_ch_t_orssi = ch_orssi
                t.opt_ch_orssi = orssi
                t.save()
            t_channels = [entry.transmitting_channel]
            slot = [entry]
            thr = entry.throughput
            ssnr = entry.self_snr
            osnr = entry.other_snr
            srssi = entry.self_rssi
            orssi = entry.other_rssi
            ch_t = entry.transmitting_channel
            ch_ssnr = entry.transmitting_channel
            ch_srssi = entry.transmitting_channel
            ch_orssi = entry.transmitting_channel
            ch_osnr = entry.transmitting_channel

        if counter % 20 == 0:
            print len(train_s)-counter, "left"
        counter += 1
    #last for loop for those elements that did not get to 'else' part
    for t in slot:
        t.opt_ch_t_thr = ch_t
        t.opt_ch_thr = thr
        t.opt_ch_t_ssnr = ch_ssnr
        t.opt_ch_ssnr = ssnr
        t.opt_ch_t_srssi = ch_srssi
        t.opt_ch_srssi = srssi
        t.opt_ch_t_osnr = ch_osnr
        t.opt_ch_osnr = osnr
        t.opt_ch_t_orssi = ch_orssi
        t.opt_ch_orssi = orssi
        t.save()
    return render_to_response('sfh/train.html', { 'trains' : train_s })


#########################################################################################
#											#
# cross(request):									#
#  - spliting the whole training set into test set and train set, check the accurcy	#
#  do it again for different test set (and so training set), repeat k times and 	#
#  average the answer; k-fold cross-validation, (by default k=10)			#
#											#
#########################################################################################

def cross(request):
    CHANNELS = [ 5.18, 5.26, 5.32, 5.5, 5.6, 5.7, 5.745, 5.785, 5.825 ]
    KEYS = [ 'other_noise', 'other_rssi', 'other_snr', 
             'self_noise',  'self_snr',   'self_rssi', 
             'throughput', 'tide_level' ]
    combs = [ list(combinations(KEYS, i)) for i in range(1,len(KEYS)+1) ]
    all_keys_combinations = [ item for sublist in combs for item in sublist ]
#    all_keys_combinations = [('tide_level'), ('tide_level', 'throughput')]
    out = []
    all_train = Train.objects.values()
    if request.method == 'GET':
        data=request.GET
        if 'nbs' in data:
            nbs = int(data.get('nbs'))
        else:
            nbs = 3
        k = 10
        print "k:", k, " nbs:", nbs         
        sets = [ [] for i in range(k) ]
        i = 0
        indices = range(len(all_train)) # list of all indecies
        while indices:
            sets[i%k].append(all_train[indices.pop(randint(0,len(indices)-1))])
            i += 1
        for keys in all_keys_combinations:
            total = 0
            total_correct = 0
            for i in range(k):
                train_s = []
                [ train_s.extend(x) for x in sets if not sets.index(x) == i ]
                correct = 0
                for test in sets[i]:
                    attributes = dict([ [x,y] for (x,y) in test.items() \
                                        if x in keys
                                      #x == 'throughput' or  
                                      #x == 'tide_level' or 
                                      #x == 'self_noise' or 
                                      #x == 'other_noise' or
                                      #x == 'other_rssi' or
                                      #x == 'self_rssi' #or
                                      #x == 'selfSNR' or
                                      #x == 'otherSNR' or
                                    ]) # 'False' because i am to lazy to comment out 'or'
                    if test.get('opt_ch_t_thr') == knn(test.get('transmitting_channel'), train_s, attributes, nbs):
                        correct+=1
                #print i,"accuracy:\t", correct/float(len(sets[i]))
                total += len(sets[i])
                total_correct += correct
            res = str(all_keys_combinations.index(keys))+" keys: "+str(keys)+" accuracy: "+str(total_correct/float(total)) 
            print res
            out.append(res)
    return render_to_response('sfh/show.html', { 'train_s' : out })

#########################################################################
#									#
# show():								#
#  - shows the whole training set					#
#									#
#########################################################################

def show(request):
    #string for arff file
#    relation = "@RELATION tide_level\n\n"
#    timestamp = "@ATTRIBUTE timestamp NUMERIC\n"
#    frequency = "@ATTRIBUTE frequency	{5.18,5.26,5.32,5.5,5.6,5.7,5.745,5.785,5.825}\n"
#    throughput = "@ATTRIBUTE throughput	NUMERIC\n"
#    tide = "@ATTRIBUTE tide	NUMERIC\n"
#    opt = "@ATTRIBUTE opt	{5.18,5.26,5.32,5.5,5.6,5.7,5.745,5.785,5.825}\n\n\n"

#    data = "@DATA\n"
#    str = relation+timestamp+frequency+throughput+tide+opt+data
    str = "abc"
    return render_to_response('sfh/show.html', { 'train_s' : str })#, 'arff' : str} )

def graph(request):
    train_s = Train.objects.all()
    channels = []
    tss=[]
    opt_ch_t=[]
    opt_ch_ssnr = []
    opt_ch_srssi = []
    opt_ch_osnr = []
    opt_ch_orssi = []
    thr_s = []
    ssnr_s = []
    srssi_s = []
    osnr_s = []
    orssi_s = []
    ts_s = []
    for tr in train_s:
        if not tr.transmitting_channel in channels:
            ts = []
            thr = []
            ssnr = []
            snoise = []
            srssi = []
            osnr = []
            onoise = []
            orssi = []
            ch = tr.transmitting_channel
            print "ch: ", ch
            channels.append(ch)
            all_data = train_s.filter(transmitting_channel=ch)
            for data in all_data:
                ts.append(int(data.timestamp % 1000000))
                thr.append(float(data.throughput))
                ssnr.append(float(data.self_snr))
                snoise.append(float(data.self_noise))
                srssi.append(float(data.self_rssi))
                osnr.append(float(data.other_snr))
                onoise.append(float(data.other_noise))
                orssi.append(float(data.other_rssi))

            #Throughput
            plt.figure()
            plt.plot(ts,thr, label=str(ch))
            plt.plot()
            plt.xlabel("Timestamp")
            plt.ylabel("Throughput")
            plt.xlim(min(ts), max(ts))
            plt.ylim(min(thr)-1, max(thr)+1)
            plt.legend(loc="upper left")
            plt.savefig("thr_" + str(ch).split(".")[0]+"_"+str(ch).split(".")[1]+".png", dpi=200)

            #Noise - self
            plt.figure()
            plt.plot(ts, snoise, label=str(ch))
            plt.plot()
            plt.xlabel("Timestamp")
            plt.ylabel("self_noise")
            plt.xlim(min(ts), max(ts))
            plt.ylim(min(snoise)-1, max(snoise)+1)
            plt.legend(loc="upper left")
            plt.savefig("snoise_"+str(ch).split(".")[0]+"_"+str(ch).split(".")[1]+".png", dpi=200)

            #Noise - other
            plt.figure()
            plt.plot(ts, onoise, label=str(ch))
            plt.plot()
            plt.xlabel("Timestamp")
            plt.ylabel("other_noise")
            plt.xlim(min(ts), max(ts))
            plt.ylim(min(onoise)-1, max(onoise)+1)
            plt.legend(loc="upper left")
            plt.savefig("onoise_"+str(ch).split(".")[0]+"_"+str(ch).split(".")[1]+".png", dpi=200)

            #SNR - self
            plt.figure()
            plt.plot(ts, ssnr, label=str(ch))
            plt.plot()
            plt.xlabel("Timestamp")
            plt.ylabel("selfSNR")
            plt.xlim(min(ts), max(ts))
            plt.ylim(min(ssnr)-1, max(ssnr)+1)
            plt.legend(loc="upper left")
            plt.savefig("ssnr_"+str(ch).split(".")[0]+"_"+str(ch).split(".")[1]+".png", dpi=200)

            #SNR - other
            plt.figure()
            plt.plot(ts, osnr, label=str(ch))
            plt.plot()
            plt.xlabel("Timestamp")
            plt.ylabel("otherSNR")
            plt.xlim(min(ts), max(ts))
            plt.ylim(min(osnr)-1, max(osnr)+1)
            plt.legend(loc="upper left")
            plt.savefig("osnr_"+str(ch).split(".")[0]+"_"+str(ch).split(".")[1]+".png", dpi=200)

            #RSSI - self
            plt.figure()
            plt.plot(ts, srssi, label=str(ch))
            plt.plot()
            plt.xlabel("Timestamp")
            plt.ylabel("selfRSSI")
            plt.xlim(min(ts), max(ts))
            plt.ylim(min(srssi)-1, max(srssi)+1)
            plt.legend(loc="upper left")
            plt.savefig("srssi_"+str(ch).split(".")[0]+"_"+str(ch).split(".")[1]+".png", dpi=200)

            #RSSI - other
            plt.figure()
            plt.plot(ts, orssi, label=str(ch))
            plt.plot()
            plt.xlabel("Timestamp")
            plt.ylabel("otherRSSI")
            plt.xlim(min(ts), max(ts))
            plt.ylim(min(orssi)-1, max(orssi)+1)
            plt.legend(loc="upper left")
            plt.savefig("orssi_"+str(ch).split(".")[0]+"_"+str(ch).split(".")[1]+".png", dpi=200)
            
            #adding results to bigger list to show them all in a single plot
            ts_s.append(ts)
            thr_s.append(thr)
            ssnr_s.append(ssnr)
            srssi_s.append(srssi)
            osnr_s.append(osnr)
            orssi_s.append(orssi)

        # adding data to plot optimal results, need to do it for rssi and snr as well
        opt_ch_t.append(tr.opt_ch_t_thr)
        opt_ch_ssnr.append(tr.opt_ch_t_ssnr)
        opt_ch_srssi.append(tr.opt_ch_t_srssi)
        opt_ch_osnr.append(tr.opt_ch_t_osnr)
        opt_ch_orssi.append(tr.opt_ch_t_orssi)
        tss.append(tr.timestamp % 1000000)

    #singel plot with all thrs
    plt.figure()
    for i in range(len(thr_s)):
        plt.plot(ts_s[i], thr_s[i], label=str(channels[i]))
    plt.xlabel("Timestamp")
    plt.ylabel("Throughput")
    plt.xlim(min(tss), max(tss))
    plt.ylim(min(min(thr_s)), max(max(thr_s)))
    plt.legend(loc="upper left")
    plt.savefig("all_thr.png", dpi=200)

    #singel plot with all self snrs
    plt.figure()
    for i in range(len(ssnr_s)):
        plt.plot(ts_s[i], ssnr_s[i], label=str(channels[i]))
    plt.xlabel("Timestamp")
    plt.ylabel("SelfSNR")
    plt.xlim(min(tss), max(tss))
    plt.ylim(min(min(ssnr_s)), max(max(ssnr_s)))
    plt.legend(loc="upper left")
    plt.savefig("all_ssnr.png", dpi=200)

    #singel plot with all self RSSI
    plt.figure()
    for i in range(len(srssi_s)):
        plt.plot(ts_s[i], srssi_s[i], label=str(channels[i]))
    plt.xlabel("Timestamp")
    plt.ylabel("selfRSSI")
    plt.xlim(min(tss), max(tss))
    plt.ylim(min(min(srssi_s)), max(max(srssi_s)))
    plt.legend(loc="upper left")
    plt.savefig("all_srssi.png", dpi=200)

    #singel plot with all other SNR
    plt.figure()
    for i in range(len(osnr_s)):
        plt.plot(ts_s[i], osnr_s[i], label=str(channels[i]))
    plt.xlabel("Timestamp")
    plt.ylabel("otherSNR")
    plt.xlim(min(tss), max(tss))
    plt.ylim(min(min(osnr_s)), max(max(osnr_s)))
    plt.legend(loc="upper left")
    plt.savefig("all_osnr.png", dpi=200)

    #singel plot with all other RSSI
    plt.figure()
    for i in range(len(orssi_s)):
        plt.plot(ts_s[i], orssi_s[i], label=str(channels[i]))
    plt.xlabel("Timestamp")
    plt.ylabel("other RSSI")
    plt.xlim(min(tss), max(tss))
    plt.ylim(min(min(orssi_s)), max(max(orssi_s)))
    plt.legend(loc="upper left")
    plt.savefig("all_orssi.png", dpi=200)

    
    #optimal channels in terms of throughput
    plt.figure()
    plt.plot(tss, opt_ch_t)
    plt.xlabel("Timestamp")
    plt.ylabel("Optimal channel/throughput")
    plt.xlim(min(tss), max(tss))
    plt.ylim(5, 6)
    plt.legend(loc="upper left")
    plt.savefig("opt_ch_thr.png", dpi=200)

    #optimal channels in terms of self SNR
    plt.figure()
    plt.plot(tss, opt_ch_ssnr)
    plt.xlabel("Timestamp")
    plt.ylabel("Optimal channel/self SNR")
    plt.xlim(min(tss), max(tss))
    plt.ylim(5, 6)
    plt.legend(loc="upper left")
    plt.savefig("opt_ch_selfSNR.png", dpi=200)

    #optimal channels in terms of self RSSI
    plt.figure()
    plt.plot(tss, opt_ch_srssi)
    plt.xlabel("Timestamp")
    plt.ylabel("Optimal channel/self RSSI")
    plt.xlim(min(tss), max(tss))
    plt.ylim(5, 6)
    plt.legend(loc="upper left")
    plt.savefig("opt_ch_selfRSSI.png", dpi=200)

    #optimal channels in terms of other SNR
    plt.figure()
    plt.plot(tss, opt_ch_osnr)
    plt.xlabel("Timestamp")
    plt.ylabel("Optimal channel/other SNR")
    plt.xlim(min(tss), max(tss))
    plt.ylim(5, 6)
    plt.legend(loc="upper left")
    plt.savefig("opt_ch_otherSNR.png", dpi=200)

    #optimal channels in terms of other RSSI
    plt.figure()
    plt.plot(tss, opt_ch_orssi)
    plt.xlabel("Timestamp")
    plt.ylabel("Optimal channel/other RSSI")
    plt.xlim(min(tss), max(tss))
    plt.ylim(5, 6)
    plt.legend(loc="upper left")
    plt.savefig("opt_ch_otherRSSI.png", dpi=200)


    return render_to_response('sfh/graph.html')


#################################################################################
#										#
#  pearson():									#
#   - function that computes the pearson correlation coefficient between every 	#
#     pair of transmitting channel in terms of throughput, SNR(self and other)	#
#     and RSSI(self and other) and prints the result in a form o table/matrix 	#
#     of 9x9 size								#
#   - the result should be seen in both browser and terminal (server log screen	#
#										#
#################################################################################

def pearson(request):
    multi_all_channels = Train.objects.values('transmitting_channel').order_by('transmitting_channel').distinct()
    CHANNELS = [ t.get('transmitting_channel') for t in multi_all_channels ]
#    for t in multi_all_channels:
#        if not t.get('transmitting_channel') in CHANNELS:
#            CHANNELS.append(t.get('transmitting_channel'))

    print CHANNELS
    all_train_s = Train.objects.values()
    slot_ch = []
    slot = []
    values = [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    for t in all_train_s:
        #building a slot
        if not t.get('transmitting_channel') in slot_ch:
            slot_ch.append(t.get('transmitting_channel'))
            slot.append(t)
        # working on a slot and initialize new slot
        else:
            for ch1 in CHANNELS:
                for ch2 in CHANNELS:
                    if ch1 in slot_ch and ch2 in slot_ch:
                        for t in slot:
                            if t.get('transmitting_channel') == ch1:
                                x = t
                            if t.get('transmitting_channel') == ch2:
                                y = t
                        values[CHANNELS.index(ch1)*len(CHANNELS)+CHANNELS.index(ch2)].append({'x':x, 'y':y})
            slot_ch = [t.get('transmitting_channel')]
            slot = [t]

    pearson_thr = [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson_osnr = [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson_ssnr = [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson_srssi = [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson_orssi = [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson_onoise = [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson_snoise = [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson_tide = [ [] for c1 in CHANNELS for c2 in CHANNELS ]

    for i in range(len(values)):
        n = len(values[i])
        # pearson throughput
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('throughput') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('throughput') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('throughput'),2) for v in values[i] ])
        pearson_thr[i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)) )
        #pearson self SNR
        sumxy = sum([ v.get('x').get('selfSNR') * v.get('y').get('selfSNR') for v in values[i] ])
        sumx = sum([ v.get('x').get('selfSNR') for v in values[i] ])
        sumy = sum([ v.get('y').get('selfSNR') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('selfSNR'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('selfSNR'),2) for v in values[i] ])
        pearson_ssnr[i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
        #pearson self RSSI
        sumxy = sum([ v.get('x').get('self_rssi') * v.get('y').get('self_rssi') for v in values[i] ])
        sumx = sum([ v.get('x').get('self_rssi') for v in values[i] ])
        sumy = sum([ v.get('y').get('self_rssi') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('self_rssi'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('self_rssi'),2) for v in values[i] ])
        pearson_srssi[i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
        #pearson other SNR
        sumxy = sum([ v.get('x').get('otherSNR') * v.get('y').get('otherSNR') for v in values[i] ])
        sumx = sum([ v.get('x').get('otherSNR') for v in values[i] ])
        sumy = sum([ v.get('y').get('otherSNR') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('otherSNR'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('otherSNR'),2) for v in values[i] ])
        pearson_osnr[i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
        #pearson other RSSI
        sumxy = sum([ v.get('x').get('other_rssi') * v.get('y').get('other_rssi') for v in values[i] ])
        sumx = sum([ v.get('x').get('other_rssi') for v in values[i] ])
        sumy = sum([ v.get('y').get('other_rssi') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('other_rssi'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('other_rssi'),2) for v in values[i] ])
        pearson_orssi[i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
        #pearson other noise to throughput
        sumxy = sum([v.get('x').get('other_noise') * v.get('y').get('throughput') for v in values[i] ])
        sumx = sum([ v.get('x').get('other_noise') for v in values[i] ])
        sumy = sum([ v.get('y').get('throughput') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('other_noise'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('throughput'),2) for v in values[i] ])
        pearson_onoise[i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
 
        #pearson self noise to throughput
        sumxy = sum([v.get('x').get('self_noise') * v.get('y').get('throughput') for v in values[i] ])
        sumx = sum([ v.get('x').get('self_noise') for v in values[i] ])
        sumy = sum([ v.get('y').get('throughput') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('self_noise'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('throughput'),2) for v in values[i] ])
        pearson_snoise[i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
  
        #pearson tide to throughput
        sumxy = sum([v.get('x').get('tide_level') * v.get('y').get('throughput') for v in values[i] ])
        sumx = sum([ v.get('x').get('tide_level') for v in values[i] ])
        sumy = sum([ v.get('y').get('throughput') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('tide_level'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('throughput'),2) for v in values[i] ])
        pearson_tide[i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))

 


    STR = ""
    for i in range(len(pearson_thr)):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson_thr[i],3)) + "\t"
    print "Pearson for throughput:" + STR
    STR = ""
    for i in range(len(pearson_ssnr)):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson_ssnr[i],3)) + "\t"
    print "Pearson for self SNR:" + STR
    STR = ""
    for i in range(len(pearson_srssi)):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson_srssi[i],3)) + "\t"
    print "Pearson for self RSSI:" + STR

    STR = ""
    for i in range(len(pearson_osnr)):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson_osnr[i],3)) + "\t"
    print "Pearson for other SNR:" + STR
    STR = ""
    for i in range(len(pearson_orssi)):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson_orssi[i],3)) + "\t"
    print "Pearson for other RSSI:" + STR

    STR = ""
    for i in range(len(pearson_onoise)):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson_onoise[i],3)) + "\t"
    print "Pearson for other noise/throughput:" + STR    

    STR = ""
    for i in range(len(pearson_snoise)):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson_snoise[i],3)) + "\t"
    print "Pearson for self noise/throughput:" + STR


    STR = ""
    for i in range(len(pearson_tide)):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson_tide[i],3)) + "\t"
    print "Pearson for tide level/throughput:" + STR


    return render_to_response('sfh/index.html')
