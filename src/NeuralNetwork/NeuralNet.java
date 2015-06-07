/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Harvey
 */
public class NeuralNet extends Classifier {
    private List<Double> mInput;            // First layer inputs
    private List<Double> mOutput;           // Inputs for the other layers 
    private List<List<Neuron>> mNetwork;    // All of the neurons
    private Instances mInstances;           
    private double mBias;                   // Variable for bias value
    private double mlRate;                  // Learning Rate
    private int mArrSize;
    private int mClassNum;
    private int mLayers;
    private int mhNeurons;                  // Number of hidden layer neurons
    private int moNeurons;                  // Number of output layer neurons
    private int mStopNum;
    
    /**
     *
     * @param pInsts
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances pInsts) throws Exception {
       mBias = -1.0;
       mInstances = pInsts;
       mArrSize = mInstances.numAttributes() - 1;
       mClassNum = mInstances.numClasses();
       mInput = new ArrayList<>();
       mOutput = new ArrayList<>();
       mNetwork = new ArrayList<>();
       mOutput.add(mBias);
       mStopNum = 50;
       mLayers = 3;
       mlRate = 0.2;
       mhNeurons = 4;       
       moNeurons = mClassNum;
       
       initializeNetwork(mInstances.firstInstance());
       
                    
    }
    
    private void initializeNetwork(Instance pInst)
    {
        double output;
        List<Double> tempArr;
        Neuron tempNeuron;
        mInput.add(mBias);

        for (int i = 0; i < (mArrSize); i++)
        {
            mInput.add(pInst.value(i));
        }

        for (int j = 0; j < mLayers; j++)
        {
            List<Neuron> neurons  = new ArrayList<>();
            if (j < mLayers - 1)
            {
               tempArr = mOutput;
               mOutput.clear();
               for (int k = 0; k <= mhNeurons; k++)
               {
                  neurons.add(new Neuron(mInput.size()));
                  if(j == 0)
                  {
                      tempNeuron = neurons.get(k);
                      output = tempNeuron.inputNeuron(mInput, mInput.size());
                      mOutput.add(output);
                  }
                  else 
                  {
                      
                      tempNeuron = neurons.get(k);
                      if (k == 0)
                      {
                          mOutput.add(mBias);
                          output = tempNeuron.inputNeuron(tempArr, tempArr.size());
                          mOutput.add(output);
                          
                      }   
                      else
                      {
                      output = tempNeuron.inputNeuron(tempArr, tempArr.size());
                      mOutput.add(output);
                      }
                  }
               }
           }
           else 
           {
               tempArr = mOutput;
               mOutput.clear();
               for (int i = 0; i <= moNeurons; i++)
               {
                   neurons.add(new Neuron(tempArr.size()));
                   
                   tempNeuron = neurons.get(i);
                   output = tempNeuron.inputNeuron(tempArr, tempArr.size());
                   mOutput.add(output);                   
               }
           }          
           
           mNetwork.add(neurons);
       }      
       
      
       mInput.clear();
       mOutput.clear();
    }
    
    private double trainNetwork(Instance pInst)
    {
        double error = 0.0;
        double temp = runNetwork(pInst);
        int cIndex = pInst.classIndex();
        double target;
        target = pInst.value(cIndex);
  
        for (int i = mLayers - 1; i >= 0; i--)
        {
            // Calculate error
            if (i == (mLayers - 1))
            {
                // i = input from layer on the left
                // j = current node
                // Output layer error
                // errj = aj(1 - aj)(aj - tj)
                for (int j = 0; j < mClassNum; j++)
                {
                   temp = mOutput.get(j);
                   if (target == (double) j)
                   {
                       error = temp * (1 - temp) * (temp - target);
                   }
                   else
                   {
                       error = temp * (1 - temp) * (temp - 0.0);
                   }
                }
                
            }
            else
            {
                // j = current node;
                // k = node from the layer on the right
                // Hidden layer error
                // errj = aj(1 - aj) * sum(wjk * errk)
            }
            
            // Update weight ij
            // wij <- wij - mlRate * errj * ai
            
            //mNetwork.set(j, neurons);
        }
        
        return error;
    }
       
    private double runNetwork(Instance pInst)
    {
        mInput.clear();
        mOutput.clear();
        double classNum = 0.0;
        double outputVal = 0.0;
        double output;
        List<Double> tempArr;
        Neuron tempNeuron;
        mInput.add(mBias);
        
        for (int i = 0; i < (mArrSize); i++)
        {
            mInput.add(pInst.value(i));
        }

        for (int j = 0; j < mLayers; j++)
        {
            List<Neuron> neurons;
            neurons = mNetwork.get(j);
            
            if (j < mLayers - 1)
            {
               tempArr = mOutput;
               mOutput.clear();            
               for (int k = 0; k <= mhNeurons; k++)
               {
                  
                  if(j == 0)
                  {
                      tempNeuron = neurons.get(k);
                      output = tempNeuron.inputNeuron(mInput, mInput.size());
                      mOutput.add(output);
                  }
                  else 
                  {
                      
                      tempNeuron = neurons.get(k);
                      if (k == 0)
                      {
                          mOutput.add(mBias);
                          output = tempNeuron.inputNeuron(tempArr, tempArr.size());
                          mOutput.add(output);
                          
                      }   
                      else
                      {
                      output = tempNeuron.inputNeuron(tempArr, tempArr.size());
                      mOutput.add(output);
                      }
                  }
               }
           }
           else 
           {
               tempArr = mOutput;
               mOutput.clear();
               for (int i = 0; i <= moNeurons; i++)
               {
                   tempNeuron = neurons.get(i);
                   output = tempNeuron.inputNeuron(tempArr, tempArr.size());
                   mOutput.add(output);                   
               }
           }          
           
           
       }
       
       for (int num = 0; num < mClassNum; num++)
       {
           if (mOutput.get(num) > outputVal)
           {
               classNum = num;
           }
       }     
      
       return classNum;
    }
    
    /**
     *
     * @param pInst
     * @return
     * @throws Exception
     */
    @Override
    public double classifyInstance(Instance pInst) throws Exception {
       return runNetwork(pInst);             
    }  
    
}

