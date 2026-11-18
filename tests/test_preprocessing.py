import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from preprocess import candidates,project_segments,align_reference

class GeometryTests(unittest.TestCase):
    def test_exact_candidates_against_brute_force(self):
        rng=np.random.default_rng(17);a=rng.normal(size=(120,2))*100;b=a+rng.normal(size=(120,2))*200;p=rng.normal(size=(20,2))*100
        ids,dist,_=candidates(p,a,b)
        brute,_=project_segments(p[:,None,:],a[None,:,:],b[None,:,:])
        np.testing.assert_allclose(dist,np.sort(brute,axis=1)[:,:8])
    def test_long_segment_midpoint_is_far_away(self):
        a=np.array([[0.,0.],[0.,3.],[0.,5.]]);b=np.array([[10000.,0.],[1.,3.],[1.,5.]])
        ids,dist,_=candidates(np.array([[0.,.1]]),a,b,k=2)
        self.assertEqual(ids[0,0],0);self.assertAlmostEqual(dist[0,0],.1)
    def test_monotone_alignment(self):
        a=np.array([[0.,0.],[10.,0.],[20.,0.]]);b=a+[10.,0.];p=np.array([[2.,1.],[9.,1.],[15.,1.],[28.,1.]])
        label,idx,residual,margin=align_reference(p,a,b,np.arange(3))
        np.testing.assert_array_equal(label,[0,0,1,2]);self.assertTrue(np.all(np.diff(idx)>=0));np.testing.assert_allclose(residual,1)
    def test_zero_length_segment(self):
        d,f=project_segments(np.array([3.,4.]),np.array([0.,0.]),np.array([0.,0.]))
        self.assertEqual(d,5);self.assertEqual(f,0)

class AlignmentOptimality(unittest.TestCase):
    def test_matches_exhaustive_monotone_paths(self):
        import itertools
        rng=np.random.default_rng(9)
        a=rng.normal(size=(4,2));b=a+rng.normal(size=(4,2));points=rng.normal(size=(5,2))
        route=np.arange(4);labels,idx,_,_=align_reference(points,a,b,route)
        d,_=project_segments(points[:,None,:],a[None,:,:],b[None,:,:])
        expected=min(sum(d[t,j]**2 for t,j in enumerate(path)) for path in itertools.combinations_with_replacement(range(4),5))
        self.assertAlmostEqual(sum(d[t,j]**2 for t,j in enumerate(idx)),expected)

if __name__=='__main__':unittest.main()
