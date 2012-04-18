from django.shortcuts import render_to_response
from sfh.models import Train, Tide, Channels
from math import fabs, sqrt, pow, sin, radians, pi, exp, log
import sys
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from itertools import combinations
from time import time

#########################################################################
# 									#
# index():								#
#  - send the data that needs to be classified in GET request:		#
#     - tch - transmitting channel 					#
#     - thr - throughput of tch						#
# -TO-DO: add other feature check from GET request			#
#									#
#########################################################################

def index(request):
    if request.method == 'GET':
        data=request.GET
        tch = thr = None
        if 'tch' in data:
            tch = data.get('tch')
        if "thr" in data:
            thr = float(data.get('thr'))
        if thr and tch:
            result = n_bayes_bins(tch, Train.objects.values(), { 'throughput' : thr })
            print "n_bayes_bins:", result
           # print "knn:\n", knn(tch, Train.objects.values(), { 'throughput' : thr})
           # return render_to_response('sfh/index.html', { 'knn' : result })
    return render_to_response('sfh/index.html')

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
        # checing if all required values are in the get request
        logic = 'ts' in data and 'tch' in data and "thr" in data and 'ssnr' in data 
        logic = logic and 'snoise' in data and 'osnr' in data and 'onoise' in data
        if logic:
            # retrieving the values
            timestamp = (data.get('ts'))
            transmitting_channel = (data.get('tch'))
            throughput = (data.get('thr'))
            self_snr = float(data.get('ssnr'))/float(100)
            self_noise = float(data.get('snoise'))/float(100)
            self_rssi = self_snr+self_noise
            other_snr = float(data.get('osnr'))/float(100)
            other_noise = float(data.get('onoise'))/float(100)
            other_rssi = other_snr+other_noise
            # preparing the data to be saved
            line = Train(timestamp=timestamp, 
                         transmitting_channel=transmitting_channel,
                         throughput=throughput,
                         self_snr=self_snr,
                         self_noise=self_noise,
                         self_rssi=self_rssi,
                         other_snr=other_snr,
                         other_noise=other_noise,
                         other_rssi=other_rssi)
            # this save is modified in models.py to discard all entries that noise or snr are 0
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
            # storing the hight value in meters 
            h = float(data.get('tide'))/100
            line = Tide(timestamp=ts, height=h)
            line.save()
            return render_to_response('sfh/tide.html', { 'tide' : line })
   return render_to_response('sfh/tide.html', {'tide' : Tide.objects.all() })

#########################################################################################
#											#
# mix(request):										#
#  - computing the height of tide between two entries in Tide table 			#   
#  - using sin interpolation:								#
#  	A*sin(B*(x - C)) + D = y 							#
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
#													  #
# dummy(request):											  #
#  - adding dummy entries so total numeber of entries is a multiply of number of channels used		  #
#    but is adding a 0 throughput is the best idea? maybe should add -1 throughput or -infinity throghput #
#  - Produce the weak results so DO NOT use it!								  #
#													  #
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
            # looking for the best thr, self/other snr, self/other rssi
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
            # saving optimal channels and their performance
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
        # printing the current status of training
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
    # attributes/features that charaterize the link, testing all possible combinations of them as a features vector
    #KEYS = [ 'other_noise', 'other_rssi', 'other_snr', 
    #         'self_noise',  'self_snr',   'self_rssi', 
    #         'throughput' ]
    # getting all possible features combinations 
    #combs = [ list(combinations(KEYS, i)) for i in range(1,len(KEYS)+1) ]
    #flatting the list of lists of tuples to list of tuples
    #all_keys_combinations = [ item for sublist in combs for item in sublist ] 
    #all_keys_combinations = [ k + ('tide_level',) for k in all_keys_combinations ]
    #all_keys_combinations.insert(0,('tide_level',) )
#    all_keys_combinations = [('tide_level'), ('tide_level', 'throughput'),('other_noise', 'tide_level'),
#                             ('self_noise', 'tide_level') ] #just for tests
    all_keys_combinations = [#('throughput'),
                             ('tide_level', 'self_noise')]#,
#                             ('throughput', 'tide_level', 'self_noise'),
#                             ('throughput', 'tide_level', 'self_noise', 'other_noise'),
#                             ('throughput', 'tide_level', 'self_noise', 'other_noise', 'self_snr'),
#                             ('throughput', 'tide_level', 'self_noise', 'other_noise', 'self_snr', 'other_snr'),
#                             ('throughput', 'tide_level', 'self_noise', 'other_noise', 'self_snr', 'other_snr', 'self_rssi'),
#                             ('throughput', 'tide_level', 'self_noise', 'other_noise', 'self_snr', 'other_snr', 'self_rssi', 'other_rssi')] # more tests


    out = []
    all_train = Train.objects.values()
    if request.method == 'GET':
        data=request.GET
        # numbr of negihbors to consider
        if 'nbs' in data:
            nbs = int(data.get('nbs'))
        else:
            nbs = 16
        # number of bins for discretization
        if 'bins' in data:
            bins = int(data.get('bins'))
        else:
            bins = 10
        # number of folds in k-fold cross validation
        if 'folds' in data:
            k = int(data.get('folds'))
        else:
            k = 10
        res = "pearson knn "  + "folds:" + str(k) + " nbs:" + str(nbs) + " bins: " + str(bins) + "\n"
        print res
        out.append(res)
        sets = [ [] for i in range(k) ]
        i = 0
        indices = range(len(all_train)) # list of all indecies
        # randomly deviding training set into k folds/set
        while indices:
            sets[i%k].append(all_train[indices.pop(randint(0,len(indices)-1))])
            i += 1
        # checking all keys/attributes from all_keys_combination list
        # for all the fold in k-fold cross validation
        for keys in all_keys_combinations:
            print (all_keys_combinations.index(keys)+1), " keys: ", keys
            res = ""
            total = 0
            total_correctKNN = 0
            total_correctNBB = 0
            total_correctNBG = 0
            total_elapsedKNN = 0
            total_elapsedNBB = 0
            total_elapsedNBG = 0
            for i in range(k):
            # making a training set - merging other, not testing folds
                train_s = []
                [ train_s.extend(x) for x in sets if not sets.index(x) == i ]
                correctKNN = 0
                correctNBB = 0
                correctNBG = 0
                elapsedKNN = 0
                elapsedNBB = 0
                elapsedNBG = 0
                for test in sets[i]:
                #keys = Channels.objects.get(channel=test.get('transmitting_channel')).features
                    attributes = dict([ [x,y] for (x,y) in test.items() if x in keys and not x in "id" ])
                # measuring the time and accuracy of knn classifier
#                    start = time()
#                    if test.get('opt_ch_t_thr') == knn(test.get('transmitting_channel'), train_s, attributes, nbs):
#                        correctKNN += 1
#                    elapsedKNN += (time() - start)
                    
                # measuring the time and accuracy of Discretize Naive Bayes classifier
                    start = time()
                    if test.get('opt_ch_t_thr') == n_bayes_bins(test.get('transmitting_channel'), train_s, attributes, bins):
                        correctNBB += 1
                    elapsedNBB += (time() - start)

                # measuring the time and accuracy of Continous Naive Bayes classifier
#                    start = time()
#                    if test.get('opt_ch_t_thr') == n_bayes_gauss(test.get('transmitting_channel'), train_s, attributes):
#                        correctNBG += 1
#                    elapsedNBG += (time() - start)
                    
                #printing the results for each fold
#                print i,"- knn accuracy:\t", correctKNN/float(len(sets[i])), " time: ", elapsedKNN
#                print i,"- nb-gauss accuracy:\t", correctNBG/float(len(sets[i])), " time: ", elapsedNBG
                print i,"- nb-bins accuracy:\t", correctNBB/float(len(sets[i])), " time: ", elapsedNBB

                total += len(sets[i])
                total_correctKNN += correctKNN
                total_correctNBG += correctNBG
                total_correctNBB += correctNBB
                total_elapsedNBB += elapsedNBB
                total_elapsedNBG += elapsedNBG
                total_elapsedKNN += elapsedKNN
            # adding final results to res variable which will be printed in the browser
#            res += str(keys) + " accuracy: " + str(total_correctKNN/float(total)) + " avg time per fold: " + str(total_elapsedKNN/float(k)) + "\n"
#            res += " nb_gauss accuracy: " + str(total_correctNBG/float(total)) + " avg time per fold: " + str(total_elapsedNBG/float(k))
            res += " nb_bins accuracy: " + str(total_correctNBB/float(total)) + " avg time per fold: " + str(total_elapsedNBB/float(k))

            out.append(res)

       # and printing final result to terminal
#            print "knn accuracy: ", total_correctKNN/float(total), " avg time per fold: ", total_elapsedKNN/float(k) 
#            print "nb_gauss accuracy: ", total_correctNBG/float(total), "avg time per fold: ", total_elapsedNBG/float(k)
            print "nb_bins accuracy: ", total_correctNBB/float(total), "avg time per fold: ", total_elapsedNBB/float(k)

    return render_to_response('sfh/show.html', { 'train_s' : out })

#################################################################################
#										#
# n_bayes_bins(tch, train_s, attributes, bins=100 ):				#
#   - tch - current transmitting channel					#
#   - train set - to be filtered with tch					#
#   - attributes - dictionary of variables charaterizing X to classify		#
#   - number of bins to use in discretization the countinous variables		#
# Discretize the countinues variables and perform Naive Bayes classification.	#
# Returns the most probable optimal channel					#
#										#
#################################################################################

def n_bayes_bins(tch, train_s, attributes=None, bins=10):
    priori = dict()
    vals_c = dict()
    #check only the same transmitting channel
    ch_train = [ t for t in train_s if t.get('transmitting_channel') == float(tch) ]
    vals = dict([ (k,()) for k in attributes ])
    min_max = dict([ (k,(x,x)) for k,x in attributes.items() ])
    for t in ch_train: # computing the priori probablity of classes 
        t_opt_ch = t.get('opt_ch_t_thr')
        if not t_opt_ch in priori.keys():
            priori[t_opt_ch] = 1 # 1 for the first occurance
            vals_c[t_opt_ch] = vals # init vals_c[t_opt_ch]
        else:
            priori[t_opt_ch] = priori.get(t_opt_ch) + 1
        #collecting values for latter discretization, for each class
        vals_c[t_opt_ch] = dict( [ ( k,vals_c[t_opt_ch].get(k) + tuple([t.get(k)]) ) for k in attributes ] )
        #mins and maxes for computing the bins size
        min_max = dict([ (k, (min(mn, t.get(k)), max(mx,t.get(k)))) for k,(mn,mx) in min_max.items() ])
    bin_size = dict([ (k, (mx-mn)/float(bins)) for k,(mn,mx) in min_max.items() ])
    priori = dict([ ( c, occ/float(len(ch_train)) ) for c,occ in priori.items() ])
    # discretization continous variables in training set
    dis_vals = dict([
                     (c,dict([ 
                              (k,tuple([int((x-min_max[k][0])/bin_size[k]) for x in xs ]) ) 
                              for k,xs in inner_attr.items() 
                             ]) )
                     for c, inner_attr in vals_c.items()
                   ])
    # discretization continous variables in test set
    dis_attributes = dict([ (k, int((x-min_max[k][0])/bin_size[k])) for k,x in attributes.items() ])
    # computing the log posterior
    p_priori = dict([
                     (c, log(p) + sum([ log((dis_vals[c][k].count(x)+1)/float(len(dis_vals[c][k])+bins))
                                           for k,x in dis_attributes.items()]))
                     for c,p in priori.items()
                    ])
    # switching keys with values and values with keys
    p_priori = dict([ (v,k) for k,v in p_priori.items() ])
    return p_priori[max(p_priori)]


#################################################################################
#										#
# n_bayes_gauss(tch, train_s, attributes):					#
#   - tch - current transmitting channel					#
#   - train set - to be filtered with tch					#
#   - attributes - dictionary of variables charaterizing X to classify		#
# Performs Naive Bayes classification on continous variables using Gaussian.	#
# Returns the most probable optimal channel					#
#										#
#################################################################################


def n_bayes_gauss(tch, train_s, attributes=None):
    priori = dict()
    vals_c = dict()
    #check only the same transmitting channel
    ch_train = [ t for t in train_s if t.get('transmitting_channel') == float(tch) ]
    vals = dict( [ (k,()) for k in attributes ] )
    for t in ch_train: # computing the priori probablity of classes 
        t_opt_ch = t.get('opt_ch_t_thr')
        if not t_opt_ch in priori.keys():
            priori[t_opt_ch] = 1 # 1 for the first occurance
            vals_c[t_opt_ch] = vals # init vals_c[t_opt_ch]
        else:
            priori[t_opt_ch] = priori.get(t_opt_ch) + 1
        #collecting values for var and mean to gausian, for each class
        vals_c[t_opt_ch] = dict( [ ( k,vals_c[t_opt_ch].get(k) + tuple([t.get(k)]) ) for k in attributes ] )

    priori = dict([ ( c, priori.get(c)/float(len(ch_train)) ) for c in priori ])
    # means and variances for every class and every variable
    means_vars = dict([ ( cs,
                         dict([ (k, (np.mean(val[k]), np.var(val[k], ddof=1))) for k in attributes ]) 
                        ) for cs, val in vals_c.items() 
                     ])
    # computing log posteriori
    p_priori = dict([ (cs, 
                       log(p) + sum([ log_g(means_vars[cs][k], attributes[k]) for k in attributes ])
#                       p * reduce(mul, [ g(means_vars[cs][k], attributes[k]) for k in attributes ] )
                      ) for cs, p in priori.items()
                    ])
    p_priori = dict([ (v,k) for k,v in p_priori.items() ])
    return p_priori[max(p_priori)]

#################################################################
#								#
# g(m_v, x):							#
#  - m_v - tuple containing mean and variance			#
#  - x - varialbe x						#
# returns the f(x;m,v) = 1/sqrt(2*pi*v) * exp(-(x-m)^2/(2*v))	#
#								#
#################################################################

def g(m_v, x):
    (mean, var) = m_v
    if var:
        return (1/sqrt(2*pi*var)) * exp((-(x-mean)**2)/(2*var))
    elif mean == x:
        return 1
    else:
        return 0

#################################################################
#								#
# log_g(m_v, x):						#
#  - m_v - tuple containing mean and variance			#
#  - x - varialbe x						#
# returns the log(f(x;m,v)) = .5log(2*pi*v) - (x-m)^2/(2*v)	#
#								#
#################################################################


def log_g(m_v, x):
    (mean, var) = m_v
    if var:
        return -0.5*log(2*pi*var) - ((x-mean)**2)/(2*var)
    else:
        return 0

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

def knn(tch, train_s, attributes=None, k):
    neighbors = dict()
    ch_train = [ t for t in train_s if t.get('transmitting_channel') == float(tch) ]
    if not k:
        k = int(sqrt(len(ch_train)))
    # list of all features to a X to be classify
    c_x=attributes.values()
    for t in ch_train:
        t_x = [ t.get(x) for x in attributes.keys() ]
        dis = dist(t_x,c_x)
        # adding small, insignificat distance to ensure that no previous value was replaced
        while(neighbors.has_key(dis)):
            dis += 0.0000000001
        # storing the all the distances and optimals channel in dictionary
        neighbors[dis] = t.get('opt_ch_t_thr')
    neighbors_items = neighbors.items()
    neighbors_items.sort()
    k_neighbors = neighbors_items[:k]
    # retrieving k nearest neighbors classes
    classes = [ k_neighbors[i][1] for i in range(k) ]
    counters = [ classes.count(classes[i]) for i in range(k) ]
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

#########################################################################
#									#
# show():								#
#  - shows the whole training set					#
#									#
#########################################################################

def show(request):
    return render_to_response('sfh/show.html', { 'train_s' : Train.objects.all() })


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
#   - the result should be seen in both browser and terminal (server log screen)#
#     - currently only in terminal
#										#
#################################################################################

def pearson(request):

    data=request.GET
    # numbr of negihbors to consider
    if 'set' in data:
        set = True
    else:
        set = False

    multi_all_channels = Train.objects.values('transmitting_channel').order_by('transmitting_channel').distinct()
    CHANNELS = [ t.get('transmitting_channel') for t in multi_all_channels ]
#    for t in multi_all_channels:
#        if not t.get('transmitting_channel') in CHANNELS:
#            CHANNELS.append(t.get('transmitting_channel'))

#    print CHANNELS
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
                        for tt in slot:
                            if tt.get('transmitting_channel') == ch1:
                                x = tt
                            if tt.get('transmitting_channel') == ch2:
                                y = tt
                        values[CHANNELS.index(ch1)*len(CHANNELS)+CHANNELS.index(ch2)].append({'x':x, 'y':y})
            slot_ch = [t.get('transmitting_channel')]
            slot = [t]
    pearson = dict()

    pearson['throughput'] =  [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson['other_snr'] =  [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson['self_snr'] =  [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson['other_noise'] =  [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson['self_noise'] =  [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson['other_rssi'] =  [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson['self_rssi'] =  [ [] for c1 in CHANNELS for c2 in CHANNELS ]
    pearson['tide_level'] =  [ [] for c1 in CHANNELS for c2 in CHANNELS ]

    for i in range(len(values)):
        n = len(values[i])
        # pearson tlevel
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('tide_level') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('tide_level') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('tide_level'),2) for v in values[i] ])
        pearson['tide_level'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)) )
        # pearson self snr
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('self_snr') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('self_snr') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('self_snr'),2) for v in values[i] ])
        pearson['self_snr'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)) )
        #pearson self noise
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('self_noise') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('self_noise') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('self_noise'),2) for v in values[i] ])
        pearson['self_noise'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)) )
        #pearson self rssi
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('self_rssi') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('self_rssi') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('self_rssi'),2) for v in values[i] ])
        pearson['self_rssi'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)) )
        # pearson throughput
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('throughput') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('throughput') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('throughput'),2) for v in values[i] ])
        pearson['throughput'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)) )
        #pearson other SNR
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('other_snr') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('other_snr') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('other_snr'),2) for v in values[i] ])
        pearson['other_snr'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
        #pearson other RSSI
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('other_rssi') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('other_rssi') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('other_rssi'),2) for v in values[i] ])
        pearson['other_rssi'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
        #pearson other noise
        sumxy = sum([ v.get('x').get('throughput') * v.get('y').get('other_noise') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('other_noise') for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('other_noise'),2) for v in values[i] ])
        pearson['other_noise'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))
 
        #pearson self noise to throughput
        sumxy = sum([v.get('y').get('self_noise') * v.get('x').get('throughput') for v in values[i] ])
        sumy = sum([ v.get('y').get('self_noise') for v in values[i] ])
        sumx = sum([ v.get('x').get('throughput') for v in values[i] ])
        sumy2 = sum([ pow(v.get('y').get('self_noise'),2) for v in values[i] ])
        sumx2 = sum([ pow(v.get('x').get('throughput'),2) for v in values[i] ])
        pearson['self_noise'][i] = (sumxy - ((sumx*sumy)/n)) / sqrt( (sumx2 - (pow(sumx,2)/n)) * (sumy2 - (pow(sumy,2)/n)))

    result = ["check the source!"]

    STR = "Pearson for throughput:"
    for i in range(len(pearson['throughput'])):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson['throughput'][i],3)) + "\t"
    result.append(STR)
    STR = "Pearson for self SNR:"
    for i in range(len(pearson['self_snr'])):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson['self_snr'][i],3)) + "\t"
    result.append(STR)
    STR = "Pearson for self RSSI:"
    for i in range(len(pearson['self_rssi'])):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson['self_rssi'][i],3)) + "\t"
    result.append(STR)
    STR = "Pearson for other SNR:"
    for i in range(len(pearson['other_snr'])):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson['other_snr'][i],3)) + "\t"
    result.append(STR)
    STR = "Pearson for other RSSI:"
    for i in range(len(pearson['other_rssi'])):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson['other_rssi'][i],3)) + "\t"
    result.append(STR)
    STR = "Pearson for other noise/throughput:"
    for i in range(len(pearson['other_noise'])):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson['other_noise'][i],3)) + "\t"
    result.append(STR)
    STR = "Pearson for self noise/throughput:"
    for i in range(len(pearson['self_noise'])):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson['self_noise'][i],3)) + "\t"
    result.append(STR)
    STR = "Pearson for tide_level/throughput:"
    for i in range(len(pearson['tide_level'])):
        if i % len(CHANNELS) == 0:
            STR += '\n'
        STR += str(round(pearson['tide_level'][i],3)) + "\t"
    result.append(STR)
#    print result
    #return the list of keys that have strong correlation coefficient
    if set:
        chs = Channels.objects.all()
        for ch in CHANNELS:
            index = CHANNELS.index(ch)*len(CHANNELS)+CHANNELS.index(ch)
            keys = "tide_level"
            for key,pear in pearson.items():
                if fabs(pear[index]) >= 0.60 and not key in "tide_level":
                    keys+=key
            print ch, ": ", keys
            if chs.filter(channel=ch):
                c = chs.get(channel=ch)
                c.feature = keys
            else:
                c = Channels(channel=ch, features=keys)
            c.save()

    return render_to_response('sfh/show.html', {'train_s' : result})

