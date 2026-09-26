import os
import unittest
from unittest.mock import patch, Mock
import finish_periodic_campaign as campaign

class SchedulerTests(unittest.TestCase):
    def test_live_process(self):
        self.assertTrue(campaign.process_running(os.getpid()))
    def test_missing_process(self):
        with patch.object(campaign, 'Path') as p:
            p.return_value.read_text.side_effect=FileNotFoundError
            self.assertFalse(campaign.process_running(123))
    def test_zombie(self):
        with patch.object(campaign, 'Path') as p:
            p.return_value.read_text.return_value='123 (worker with spaces) Z 1 0 0'
            self.assertFalse(campaign.process_running(123))
    def test_running_and_sleeping(self):
        for state in ['R','S','D']:
            with patch.object(campaign, 'Path') as p:
                p.return_value.read_text.return_value=f'123 (worker) {state} 1 0 0'
                self.assertTrue(campaign.process_running(123))

if __name__=='__main__':unittest.main()
