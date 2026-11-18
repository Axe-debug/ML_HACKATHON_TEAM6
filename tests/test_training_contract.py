import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from model_training import score_predictions
class MetricsTests(unittest.TestCase):
    def data(self):
        return dict(y=np.array([[1,0],[0,0],[0,1]]),candidate_arc_ids=np.array([[1,2],[3,4],[1,2]]),reference_arc_id=np.array([1,5,2]),trip_id=np.array(['a','a','b']),label_reliable=np.array([True,False,True]))
    def test_misses_are_retained(self):
        metrics,pred=score_predictions(self.data(),np.array([[.9,.1],[.8,.2],[.1,.9]]))
        self.assertAlmostEqual(metrics['all_points']['top1_accuracy'],2/3)
        self.assertAlmostEqual(metrics['all_points']['candidate_miss_rate'],1/3)
        self.assertEqual(metrics['reliable_points']['top1_accuracy'],1)
        self.assertTrue(np.isfinite(metrics['all_points']['categorical_log_loss_with_miss_state']))
        self.assertGreater(metrics['all_points']['categorical_log_loss_with_miss_state'],8)
    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):score_predictions(self.data(),np.full((3,2),np.nan))
if __name__=='__main__':unittest.main()
