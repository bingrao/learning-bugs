package org.ucf.ml
package parallel


import java.util.concurrent.{Executors, ExecutorService}

class Master (configPath:String = "src/main/resources/application.conf") extends utils.Common {

  /* Load configurations from a file*/
  private val config = new Config(configPath)
  def getConfig = this.config
  private val nums_worker = this.getConfig.getNumsWorker
  private val isParallel = this.getConfig.getIsParallel


  // Create a pool of executor computing resources
  val pools: ExecutorService = Executors.newFixedThreadPool(nums_worker)

  // Load data idioms
  private val project_idioms = readIdioms(getConfig.getIdiomsPath)

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
      worker_id = index)
  }

  /* Submit workers to executors and start them*/
  def run() = if (isParallel) {
    workers.foreach(w => pools.submit(w))
    pools.shutdown()
  } else {
    workers.foreach(_.call())
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
