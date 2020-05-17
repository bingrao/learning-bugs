package org.ucf.ml
package parallel


import java.io.{FileNotFoundException, IOException}
import java.util.concurrent.{ExecutorService, Executors}
import scala.collection.JavaConversions._

/**
 * https://dzone.com/articles/java-concurrency-multi-threading-with-executorserv
 * https://www.baeldung.com/java-executor-wait-for-threads
 * @param configPath
 */
class Master (configPath:String = "src/main/resources/application.conf") extends utils.Common {

  /* Load configurations from a file*/
  private val config = new Config(configPath)
  def getConfig = this.config
  private val nums_worker = this.getConfig.getNumsWorker
  private val isParallel = this.getConfig.getIsParallel
  private var pools: ExecutorService = null

  /* Submit workers to executors and start them*/
  def run() = {
    try {

      // Load data idioms
      val project_idioms = readIdioms(getConfig.getIdiomsPath)

      /* Load buggy and target files, and save their path as a list of string*/
      val (buggy_files, fixed_files) = loadAndCheckData(getConfig.getRawBuggyFilesDir,
        getConfig.getRawFixedFilesDir)


      val total_files_nums = math.min(buggy_files.size, fixed_files.size)

      // Calcuate nums of files would be processed by a worker
      val batch_size = total_files_nums / nums_worker + 1

      // Create workers with allocated data
      val workers  = for (index <- 0 until nums_worker) yield {

        val start = index*batch_size
        val end = if (index == nums_worker - 1) total_files_nums else (index + 1) * batch_size

        new Worker(src_batch = buggy_files.slice(start, end),
          tgt_batch = fixed_files.slice(start, end),
          idioms=project_idioms,
          worker_id = index,
          granularity = METHOD)
      }


      val results = if (isParallel) {
        // Create a pool of executor computing resources
        pools = Executors.newFixedThreadPool(nums_worker)
        val data = pools.invokeAll(workers) // submit all jobs and wait them finshed
        data.map(_.get())
      } else {
        workers.map(_.call())
      }

      val buggy_abstract = results.map(_.get_buggy_abstract).mkString(EmptyString)
      val fixed_abstract = results.map(_.get_fixed_abstract).mkString(EmptyString)



      write(getConfig.getOutputBuggyDir+"buggy.txt", buggy_abstract)
      write(getConfig.getOutputBuggyDir+"fixed.txt", fixed_abstract)


    } catch  {
      case e: FileNotFoundException => {
        e.printStackTrace()
        println("Couldn't find that file.")
      }
      case e: IOException => {
        e.printStackTrace()
        println("Had an IOException trying to read that file")
      }
      case e:Exception => {
        e.printStackTrace()
      }
    } finally {
      if (pools != null) pools.shutdown()
    }

  }

  /*################################# Helper Functions #####################################*/
  def loadAndCheckData(srcPath:String, tgtPath:String):(List[String], List[String]) = {
    val srcFiles = getListOfFiles(srcPath)
    val tgtFiles = getListOfFiles(tgtPath)

    if (srcFiles.size != tgtFiles.size){
      logger.error(f"The sizes of source (${srcFiles.size}) and target (${tgtFiles}) do not match ...")
      System.exit(-1)
    }

    val files = (srcFiles zip tgtFiles).filter{
      case (src, tgt) => src.getName != tgt.getName}

    if (!files.isEmpty){
      logger.error("The input source and target files does not match ...")
      files.foreach(println _)
      System.exit(-1)
    }

    (srcFiles.map(_.getPath), tgtFiles.map(_.getPath))
  }
}
