package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type Perceptron struct {
	weights []float64 // slice of n features
	bias    float64
	n       int // number of features
}

// Initialize perceptron with random weights and bias
func NewPerceptron(n int) *Perceptron {
	weights := make([]float64, n)
	for i := range weights {
		weights[i] = rand.Float64()*0.01 - 0.005 // Small random weights
	}
	return &Perceptron{
		weights: weights,
		bias:    rand.Float64()*0.01 - 0.005,
		n:       n,
	}
}

func sigmoid(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}

func (p *Perceptron) predict(X [][]float64) []float64 {
	m := len(X)
	yHat := make([]float64, m)
	for i := 0; i < m; i++ {
		z := p.bias
		for j := 0; j < p.n; j++ {
			z += X[i][j] * p.weights[j]
		}
		yHat[i] = sigmoid(z)
	}
	return yHat
}

// Compute cross-entropy loss
func computeLoss(y, yHat []float64) float64 {
	m := len(y)
	sum := 0.0
	for i := 0; i < m; i++ {
		// Avoid log(0) by clipping
		yHatI := math.Max(1e-10, math.Min(1-1e-10, yHat[i]))
		sum += y[i]*math.Log(yHatI) + (1-y[i])*math.Log(1-yHatI)
	}
	return -sum / float64(m)
}

// Compute gradients
// gradW = 1/m * (sum for i=1 to m: (yHat[i] - y[i]) * x[i][j])) for weights
// gradB = 1/m * (sum for i=1 to m: (yHat[i] - y[i])) for bias
func (p *Perceptron) computeGradients(X [][]float64, y, yHat []float64) ([]float64, float64) {
	m := len(X)
	gradW := make([]float64, p.n)
	gradB := 0.0
	for j := 0; j < p.n; j++ {
		sumW := 0.0
		for i := 0; i < m; i++ {
			sumW += (yHat[i] - y[i]) * X[i][j]
		}
		gradW[j] = sumW / float64(m)
	}
	for i := 0; i < m; i++ {
		gradB += (yHat[i] - y[i])
	}
	gradB /= float64(m)
	return gradW, gradB
}

// Train the perceptron
func (p *Perceptron) Train(X [][]float64, y []float64, epochs int, lr float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		
		yHat := p.predict(X)
		
		loss := computeLoss(y, yHat)
		
		gradW, gradB := p.computeGradients(X, y, yHat)
		
		//Updating of the weights and bias
		for j := 0; j < p.n; j++ {
			p.weights[j] -= lr * gradW[j]
		}
		p.bias -= lr * gradB
		
		// Print progress every 10 epochs
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %.6f\n", epoch, loss)
			fmt.Printf("Accuracy: %.2f%%\n", p.Evaluate(X, y))
		}
	}
}


func (p *Perceptron) Evaluate(X [][]float64, y []float64) float64 {
	yHat := p.predict(X)
	correct := 0
	for i := range y {
		pred := 0.0
		// 0.6 is a threshold
		if yHat[i] >= 0.6 {
			pred = 1.0
		}
		if pred == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y)) * 100
}

func readCSV(filename string, n int) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return nil, nil, err
	}
	if len(header) != n+1 {
		return nil, nil, fmt.Errorf("expected %d columns, got %d", n+1, len(header))
	}

	var X [][]float64
	var y []float64
	for {
		record, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return nil, nil, err
		}
		if len(record) != n+1 {
			continue
		}

		features := make([]float64, n)
		for j := 0; j < n; j++ {
			val, err := strconv.ParseFloat(record[j], 64)
			if err != nil {
				return nil, nil, err
			}
			features[j] = val
		}
		label, err := strconv.ParseFloat(record[n], 64)
		if err != nil {
			return nil, nil, err
		}
		X = append(X, features)
		y = append(y, label)
	}
	return X, y, nil
}

func main() {
	
	n := 30
	epochs := 100
	lr := 0.1
	filename := "data.csv"

	// Read data
	fmt.Println("Reading data from", filename)
	X, y, err := readCSV(filename, n)
	if err != nil {
		fmt.Printf("Error reading CSV: %v\n", err)
		return
	}
	m := len(X)
	fmt.Printf("Loaded %d examples\n", m)

	// Split into train and test (80-20)
	trainSize := int(0.8 * float64(m))
	XTrain := X[:trainSize]
	yTrain := y[:trainSize]
	XTest := X[trainSize:]
	yTest := y[trainSize:]

	p := NewPerceptron(n)
	fmt.Println("Training perceptron...")
	start := time.Now()
	p.Train(XTrain, yTrain, epochs, lr)
	fmt.Printf("Training time: %v\n", time.Since(start))

	// Evaluate
	testAcc := p.Evaluate(XTest, yTest)
	fmt.Printf("Test Accuracy: %.2f%%\n", testAcc)
}