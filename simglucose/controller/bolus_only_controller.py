from simglucose.controller.base import Controller
from simglucose.controller.base import Action
import numpy as np
import pandas as pd
import pkg_resources
import logging

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('simglucose',
                                                'params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class BolusController(Controller):
    """
    This is a Bolus-only Controller for a Type-1
    Diabetes patient. 
    """

    def __init__(self, target=140, cr=1.0, cf=1.0):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target
        self.cr = cr
        self.cf = cf


    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')  # unit: g/min

        action = self._bb_policy(pname, meal, observation.CGM, sample_time)
        return action

    def _bb_policy(self, name, meal, glucose, env_sample_time):
        """
        Helper function to compute the basal and bolus amount.
        The basal insulin is based on the insulin amount to keep the blood
        glucose in the steady state when there is no (meal) disturbance.
               basal = u2ss (pmol/(L*kg)) * body_weight (kg) / 6000 (U/min)

        The bolus amount is computed based on the current glucose level, the
        target glucose level, the patient's correction factor and the patient's
        carbohydrate ratio.
               bolus = ((carbohydrate / carbohydrate_ratio) +
                       (current_glucose - target_glucose) / correction_factor)
                       / sample_time
        NOTE the bolus computed from the above formula is in unit U. The
        simulator only accepts insulin rate. Hence the bolus is converted to
        insulin rate.
        """

        basal = 0
        if meal > 0:
            logger.info('Calculating bolus ...')
            logger.info(f'Meal = {meal} g/min')
            logger.info(f'glucose = {glucose}')
            # bolus = (
            #         (meal * env_sample_time) / self.cr + (glucose > 150) *
            #         (glucose - self.target) / self.cf)  # unit: U
            bolus = (glucose - self.target)/self.cf + (meal*env_sample_time)/self.cr
        else:
            bolus = 0  # unit: U

        # This is to convert bolus in total amount (U) to insulin rate (U/min).
        # The simulation environment does not treat basal and bolus
        # differently. The unit of Action.basal and Action.bolus are the same
        # (U/min).
        bolus = bolus / env_sample_time  # unit: U/min
        # print(f"meal, glucose, env_sample_time: {meal}, {glucose}, {env_sample_time}")
        # print(f"basal, bolus: {basal}, {bolus}")
        return Action(basal=basal, bolus=bolus)

    def reset(self):
        pass