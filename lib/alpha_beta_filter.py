__author__ = "Philipp Koncar"
__version__ = "0.0.1"
__email__ = "p.koncar@student.tugraz.at"
__status__ = "Development"

import datetime


class AlphaBetaFilter():
    def __init__(self, alpha, beta, x_estimate):
        self.alpha = alpha
        self.beta = beta
        self.x_estimate = x_estimate
        self.v_estimate = 0
        self.filtered_activity = [x_estimate]
        self.debug_msg("Set up filter with: alpha = " + str(self.alpha) + ", beta = " + str(self.beta) +
                       " and x_estimate = " + str(self.x_estimate))

    def filter(self, measured_value, dt):
        xk = self.x_estimate + (dt * self.v_estimate)
        vk = self.v_estimate
        residual_error = measured_value - xk
        xk = xk + self.alpha * residual_error
        vk = vk + self.beta / dt * residual_error
        self.x_estimate = xk
        self.v_estimate = vk
        self.filtered_activity.append(round(self.x_estimate))

    @staticmethod
    def debug_msg(msg):
        print "  \x1b[31m-ABF-\x1b[00m [\x1b[36m{}\x1b[00m] \x1b[31m{}\x1b[00m".format(datetime.datetime.now().strftime("%H:%M:%S"), msg)