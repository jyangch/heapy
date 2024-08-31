import time
import subprocess as sp


def doing(cmd, logfile, dev=True):
    sp.call("echo '\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>' | tee -a " + logfile, shell=True)
    sp.call("(echo 'current dir: ' && pwd) | tee -a " + logfile, shell=True)
    sp.call("echo 'I am doing:' | tee -a " + logfile, shell=True)
    sp.call("echo '<" + cmd + ">' | tee -a " + logfile, shell=True)
    sp.call("echo 'Start: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "' | tee -a " + logfile, shell=True)
    if dev: msg = sp.call('(' + cmd + " && echo 'True'" + ')' + ' 2>&1 | tee -a ' + logfile, shell=True)
    else: msg = sp.call('(' + cmd + " && echo 'True'" + ')' + ' >> ' + logfile + ' 2>&1', shell=True)

    with open(logfile) as file_object:
        lines = file_object.readlines()
    if msg == 0 and lines[-1].rstrip() == 'True':
        sp.call("echo 'Stop: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "' | tee -a " + logfile, shell=True)
        sp.call("echo 'I have done successfully!' | tee -a " + logfile, shell=True)
        sp.call("echo '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<' | tee -a " + logfile, shell=True)
    elif msg != 0 or lines[-1].rstrip() != 'True':
        sp.call("echo 'Stop: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "' | tee -a " + logfile, shell=True)
        sp.call("echo 'Something is WRONG!' | tee -a " + logfile, shell=True)
        sp.call("echo '============================================================' | tee -a " + logfile, shell=True)
