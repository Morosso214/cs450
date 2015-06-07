/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import java.util.List;
import java.util.Random;

/**
 *
 * @author Harvey
 */
public class Neuron {

    private double[] mWeight;
    /**
     *
     * @param in
     * @param size
     * @return 
     */
    Neuron(int size)
    {
        mWeight = new double[size];
        for (int i = 0; i < size; i++)
            mWeight[i] = randNum();
    }
    
    Neuron(double[] weight)
    {
        mWeight = weight;
    }
    
    public double[] getWeights()
    {
       return mWeight;
    }
    
    public void setWeights(double[] weight)
    {
        mWeight = weight;
    }
    
    public double inputNeuron(List<Double> in, int size)
    {          
        double sum = 0;
        double val;
        // Move to constructor
        
        
        for (int j = 0; j < size; j++)
            sum += in.get(j) * mWeight[j]; 
        val = -1 * Math.exp(sum);
        val = 1 / (1 + val);
        
        
        return val;
    }
    
    private double randNum()
    {
        Random rand = new Random();
        double rNum = rand.nextDouble() * 2 - 1;
        return rNum;
    }
}
