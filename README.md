# Transformer-pytorch
A PyTorch implementation of Transformer in "Attention is All You Need" (https://arxiv.org/abs/1706.03762)

This repo focuses on clean, readable, and modular implementation of the paper.

<img width="559" alt="screen shot 2018-09-27 at 1 49 14 pm" src="https://user-images.githubusercontent.com/2340721/46123973-44b08900-c25c-11e8-9468-7aef9e4e3f18.png">

## Requirements
- Python 3.7+
- [PyTorch 1.5+](http://pytorch.org/)
- [NumPy](http://www.numpy.org/)
- [NLTK](https://www.nltk.org/)
- [tqdm](https://github.com/tqdm/tqdm)

```
# conda create -n learning_fix python=3.7
# conda activate learning_fix
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# conda install -c pytorch torchtext
```
## Usage

### Prepare datasets
This repo comes with example data in `data/` directory. To begin, you will need to prepare datasets with given data as follows:
```
$ python datasets.py --train_source=data/example/raw/src-train.txt --train_target=data/example/raw/tgt-train.txt --val_source=data/example/raw/src-val.txt --val_target=data/example/raw/tgt-val.txt --save_data_dir=data/example/processed
OR
$ sh datasets.sh
```

The example data is brought from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
The data consists of parallel source (src) and target (tgt) data for training and validation.
A data file contains one sentence per line with tokens separated by a space.
Below are the provided example data files.

- `src-train.txt`
- `tgt-train.txt`
- `src-val.txt`
- `tgt-val.txt`

### Train model
To train model, provide the train script with a path to processed data and save files as follows:

```
$ python train.py --data_dir=data/example/processed --save_config=checkpoints/example_config.json --save_checkpoint=checkpoints/example_model.pth --save_log=logs/example.log 
OR
$ sh train.sh
```

This saves model config and checkpoints to given files, respectively.
You can play around with hyperparameters of the model with command line arguments. 
For example, add `--epochs=300` to set the number of epochs to 300. 

### Translate
To translate a sentence in source language to target language:
```
$ python predict.py --source="There is an imbalance here ." --config=checkpoints/example_config.json --checkpoint=checkpoints/example_model.pth
OR
$ sh predict.sh
Candidate 0 : Hier fehlt das Gleichgewicht .
Candidate 1 : Hier fehlt das das Gleichgewicht .
Candidate 2 : Hier fehlt das das das Gleichgewicht .
```

It will give you translation candidates of the given source sentence.
You can adjust the number of candidates with command line argument. 

### Evaluate
To calculate BLEU score of a trained model:
```
$ python evaluate.py --save_result=logs/example_eval.txt --config=checkpoints/example_config.json --checkpoint=checkpoints/example_model.pth

BLEU score : 0.0007947

OR
$ sh evaluate.sh
```

## File description
- `models.py` includes Transformer's encoder, decoder, and multi-head attention.
- `embeddings.py` contains positional encoding.
- `losses.py` contains label smoothing loss.
- `optimizers.py` contains Noam optimizer.
- `metrics.py` contains accuracy metric.
- `beam.py` contains beam search.
- `datasets.py` has code for loading and processing data. 
- `train.py` trains model.
- `predict.py` translates given source sentence with a trained model.
- `evaluate.py` calculates BLEU score of a trained model.

## Reference
- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Transformer Pytorch](https://github.com/dreamgonfly/Transformer-pytorch)

## Author
[@Bing](https://github.com/bingrao)


You can use following maven commands to run your application.
#Run Java Program
```
mvn exec:java -Dexec.mainClass="com.jhh.Main"  
mvn exec:java -Dexec.mainClass="akka.Main" -Dexec.args="com.jhh.HelloWorld" 
```

#Run Scala Program
```
mvn scala:run -DmainClass=com.jhh.App -DaddArgs="arg1|arg2|arg3"
```

#Assembly all project into a single jar file 
```
mvn clean compile assembly:single

or

mvn clean compile package
```
After the above command, there are two jar files would be generated: 
1. a jar file with all depandent packages
2. a single jar file only containing the project source code

#Change main and premain class file in pom file
You need to revise the pom file to specify a new main/premain 
class entry if you want to a new one in your code. The default
value is "org.ucf.ml.ScalaApp"

There are two properties in the pom file
1. **project.main.class**:  Set the main entry  of all project 
2. **project.main.class**:  Set the premain entry of Java instrumenttion


# Execute jar file
```
java -cp {all needed depandency package } -jar {the target jar file} 
```
For example 

```
$ pwd
/Users/Bing/Documents/workspace/test
$ ls
README.MD	pom.xml		src		target		test.iml
$ java -jar target/test-1.0-SNAPSHOT-jar-with-dependencies.jar 
Hello World from Scala
```

In the pom file,  "{package}.ScalaApp" is set as the main entry class of the project. 
If you want to switch to "{package}.JavaApp", you just modify pom file