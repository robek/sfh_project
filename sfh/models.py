from django.db import models

# Database:
# Train :
# ____________________________________________________________________________________________________________
# |             |              |             |             |             |      |      |       |      |      |
# | timestamp   | transmitting | throughput  | tide        | self        | self | self | other | other| other|
# |             | channel      |             | level       | snr         | noise| rssi | snr   | noise| rssi |
# |             |              |             |             |             |      |      |       |      |      |
# ------------------------------------------------------------------------------------------------------------
# | optimal     | optimal      | optimal     | optimal     | optimal     |
# | channel /   | channel /    | channel /   | channel /   | channel /   |
# | throughput  | self snr     | self rssi   | other snr   | other rssi  |
# ------------------------------------------------------------------------
# | throughput  | self snr     | self rssi   | other snr   | other rssi  |
# | for optimal | for optimal  | for optimal | for optimal | for optimal |
# | channel     | channel      | channel     | channel     | channel     |
# ------------------------------------------------------------------------
class Train(models.Model):
    timestamp = models.IntegerField()

    transmitting_channel = models.FloatField()

    throughput = models.FloatField()

    self_snr = models.FloatField()
    self_noise = models.FloatField(null=True)
    self_rssi = models.FloatField(null=True)

    other_snr = models.FloatField()
    other_noise = models.FloatField(null=True)
    other_rssi = models.FloatField(null=True)

    tide_level = models.FloatField(null=True)

    opt_ch_t_thr = models.FloatField(null=True)
    opt_ch_thr = models.FloatField(null=True)
    opt_ch_t_ssnr = models.FloatField(null=True)
    opt_ch_ssnr = models.FloatField(null=True)
    opt_ch_t_srssi = models.FloatField(null=True)
    opt_ch_srssi = models.FloatField(null=True)
    opt_ch_t_osnr = models.FloatField(null=True)
    opt_ch_osnr = models.FloatField(null=True)
    opt_ch_t_orssi = models.FloatField(null=True)
    opt_ch_orssi = models.FloatField(null=True)

    # printing
    def __unicode__(self):
        ret = str(self.timestamp) + " " + str(self.transmitting_channel) + " " + str(self.throughput) + " "
        ret += str(self.self_snr) + " " + str(self.self_noise) + " " 
        ret += str(self.self_rssi) + " "
        ret += str(self.other_snr) + " " + str(self.other_noise) + " "
        ret += str(self.other_rssi)
        if self.tide_level:
            ret += " " + str(self.tide_level)
        if self.opt_ch_t_thr and self.opt_ch_thr:
            ret += " " + str(self.opt_ch_t_thr) 
            ret += " " + str(self.opt_ch_thr)
        if self.opt_ch_t_ssnr and self.opt_ch_ssnr:
            ret += " " + str(self.opt_ch_t_ssnr) 
            ret += " " + str(self.opt_ch_ssnr)
        if self.opt_ch_t_srssi and self.opt_ch_srssi:
            ret += " " + str(self.opt_ch_t_srssi) 
            ret += " " + str(self.opt_ch_srssi)
        if self.opt_ch_t_osnr and self.opt_ch_osnr:
            ret += " " + str(self.opt_ch_t_osnr)
            ret += " " + str(self.opt_ch_osnr)
        if self.opt_ch_t_orssi and self.opt_ch_orssi:
            ret += " " + str(self.opt_ch_t_orssi)
            ret += " " + str(self.opt_ch_orssi)
        return ret

    def save(self, *args, **kwargs):
        if self.self_snr and self.self_noise and self.other_snr and self.other_noise:
            super(Train, self).save(*args, **kwargs)

    class Meta:
        ordering = ['timestamp']

class Tide(models.Model):
    timestamp=models.IntegerField()
    height=models.FloatField()

    def __unicode__(self):
        return str(self.timestamp) + " " + str(self.height)

    class Meta:
        ordering = ['timestamp']
