import calendar
import urllib
import time

start=1330014051
end=1330276609
timestamp_tide = []
while start <= end:
    t_time = time.gmtime(start)
    if t_time.tm_mon >= 10:
        date_to_url = str(t_time.tm_year)+str(t_time.tm_mon)+str(t_time.tm_mday)
    else:
        date_to_url = str(t_time.tm_year)+str(0)+str(t_time.tm_mon)+str(t_time.tm_mday)
    url = "http://www.tidetimes.org.uk/mallaig-tide-times-" + date_to_url
    print url
    f = urllib.urlopen(url)
    page = f.readlines()
    f.close()
    # interesting stuff are only in both <table> and <span> tags;
    # if page structure changes this should be change as well
    state = { 'table' : 0, 'span' : 0 }
    info = []
    for l in page:
        if "<span>" in l:
            state['span'] = 1
        else:
            state['span'] = 0
        if "<table id=\"tide_summary\">" in l:
            state['table'] = 1
        if state['table'] and state['span']:
            info.append(l)
        if "</table>" in l:
            state['table'] = 0
    HM_tide = [ 
                 ( l[l.find("<span>")+len("<span>") : l.find("</span>")],
                   l[l.find("(")+len("(") : l.find("m)")] ) 
                   for l in info 
                ]
    time_string = str(t_time.tm_mday) + " " + str(t_time.tm_mon) + " " + str(t_time.tm_year) + " "
    timestamp_tide.extend([ (calendar.timegm(time.strptime(time_string + str(ctime), "%d %m %Y %H:%M")), tide)
                          for ctime,tide in HM_tide ])
    start += 86400

for timestamp, tide in timestamp_tide:
    print timestamp, tide
