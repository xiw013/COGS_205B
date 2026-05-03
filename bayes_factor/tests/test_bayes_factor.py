import unittest
from bayes_factor import BayesFactor
import numpy as np

class TestBayesFactor(unittest.TestCase):
    def setUp(self):
        self.bayes_factor1 = BayesFactor(5, 2)
        
    #-----------------------------
    # Testing for the Constructor 
    #-----------------------------

    # testing if the method returns expected data type/shape
    # testing if the method is callable
    def test_valid_constructor_values(self):
        b1 = BayesFactor(6, 6)
        self.assertEqual((b1.n, b1.k), (6, 6))
        
        b2 = BayesFactor(10, 0)
        self.assertEqual((b2.n, b2.k), (10, 0))

        b3 = BayesFactor(0, 0)
        self.assertEqual((b3.n, b3.k), (0, 0))

        b4 = BayesFactor(10, 10)
        self.assertEqual((b4.n, b4.k), (10, 10))


    def test_invalid_constructor_values(self):

        # testing if the input contains string
        with self.assertRaises(ValueError) as exception_context1:
            b = BayesFactor("no", 1)
        self.assertEqual(
            str(exception_context1.exception),
            "All input must be a non-negative integer."
        )

        # testing if the input contains bool
        with self.assertRaises(ValueError) as exception_context2: 
            b = BayesFactor(0, True)
        self.assertEqual(
            str(exception_context2.exception),
            "All input must be a non-negative integer."
        )

        # testing if any of the input is negative
        with self.assertRaises(ValueError) as exception_context3: 
            b = BayesFactor(-1, -3)
        self.assertEqual(
            str(exception_context3.exception),
            "All input must be a non-negative integer."
        )

        with self.assertRaises(ValueError) as exception_context4: 
            b = BayesFactor(0, -5)
        self.assertEqual(
            str(exception_context4.exception),
            "All input must be a non-negative integer."
        )

        # testing if the k > n
        with self.assertRaises(ValueError) as exception_context5: 
            b = BayesFactor(6, 7)
        self.assertEqual(
            str(exception_context5.exception),
            "K cannot be greater than N."
        )



    #---------------------------------
    # Testing for likelihood function
    #---------------------------------
    
    # testing if the method returns expected data type/shape
    # testing if the method is callable
    def test_valid_likelihood_input(self):
        l1 = self.bayes_factor1.likelihood(0.5)
        self.assertEqual(l1, 0.3125)

        l2 = self.bayes_factor1.likelihood(0.1)
        self.assertAlmostEqual(l2, 0.0729)

        l3 = self.bayes_factor1.likelihood(1)
        self.assertEqual(l3, 0)

        l4 = self.bayes_factor1.likelihood(0)
        self.assertEqual(l4, 0)

    # test invalid theta values
    def test_invalid_likelihood_theta(self):
        # When theta is negative
        with self.assertRaises(ValueError) as exception_context: 
            self.bayes_factor1.likelihood(-0.3)
        self.assertEqual(
            str(exception_context.exception),
            "The input theta has to be between 0 and 1 (inclusive)."
        )

        # When theta is greater than 1
        with self.assertRaises(ValueError) as exception_context: 
            self.bayes_factor1.likelihood(1.2)
        self.assertEqual(
            str(exception_context.exception),
            "The input theta has to be between 0 and 1 (inclusive)."
        )

        # when theta is a string
        with self.assertRaises(ValueError) as exception_context:
            self.bayes_factor1.likelihood("hello")
        self.assertEqual(
            str(exception_context.exception),
            "The input theta need to be a number."
        )

    #------------------------------------
    # Testing for evidence_slab function
    #------------------------------------

    # testing if the method returns expected data type/shape
    # testing if the method is callable
    def test_evidence_slab_normal(self):
        result = self.bayes_factor1.evidence_slab()
        self.assertAlmostEqual(result, 1/6)


    def test_evidence_slab_edge(self):
        # testing for k = 0
        l1 = BayesFactor(10, 0)
        result1 = l1.evidence_slab()
        self.assertEqual(result1, 1/11)

        # testing for n = k
        l2 = BayesFactor(10, 10)
        result2 = l1.evidence_slab()
        self.assertEqual(result2, 1/11)

        # testing for n = 1000, k = 500
        l3 = BayesFactor(1000, 500)
        result3 = l3.evidence_slab()
        # intentionally failling?
        self.assertAlmostEqual(result3, 1/1001, places = 8)

    #-------------------------------------
    # Testing for evidence_spike function
    #-------------------------------------

    # testing if the method returns expected data type/shape
    # testing if the method is callable
    def test_evidence_spike_normal(self):
        result = self.bayes_factor1.evidence_spike()
        self.assertEqual(result, 0.3125)

    def test_evidence_spike_edge(self):
        # testing for k = 0
        l1 = BayesFactor(10, 0)
        result1 = l1.evidence_spike()
        self.assertEqual(result1, (1-0.5)**10)

        # testing for n = k
        l2 = BayesFactor(10, 10)
        result2 = l1.evidence_spike()
        self.assertEqual(result2, (0.5)**10)

        # testing for n = 1000, k = 500
        l3 = BayesFactor(1000, 500)
        result3 = l3.evidence_spike()
         # intentionally failling?
        self.assertAlmostEqual(result3, 0.0252)
    
    #-----------------------------------
    # Testing for bayes_factor function 
    #-----------------------------------
    
    # testing if the method returns expected data type/shape
    # testing if the method is callable
    def test_bayes_factor_basic(self):
        result = self.bayes_factor1.bayes_factor()
        self.assertAlmostEqual(result, (1/6)/0.3125)

    # 
    def test_bayes_factor_edge_cases(self):
        # testing for k = 0
        l1 = BayesFactor(10, 0)
        result1 = l1.bayes_factor()
        self.assertEqual(result1, (1/11) / ((1-0.5)**10))

        # testing for n = k
        l2 = BayesFactor(10, 10)
        result2 = l1.bayes_factor()
        self.assertEqual(result1, (1/11) / ((1-0.5)**10))

        # testing for n = 1000, k = 500
        l3 = BayesFactor(1000, 500)
        result3 = l3.bayes_factor()
         # intentionally failling?
        self.assertAlmostEqual(result3,(1/1001)/0.0252, places = 4)
        
        


    

