package org.ucf.ml
package parallel


import java.nio.file.{Files, Paths}
import java.util.stream.Collectors
import scala.collection.mutable
import scala.collection.JavaConversions._
import java.io.File
class Master (configPath:String = "src/main/resources/application.conf",
              nums_worker:Int = 3) extends utils.Common {

  /* Load configurations from a file*/
  private val config = new Config(configPath)
  def getConfig = this.config
  private val project_idioms = readIdioms(getConfig.getIdiomsPath)



  val buggy_files:List[String] = getListOfFiles(getConfig.getRawBuggyFilesDir).map(_.getPath)
  val fixed_files:List[String] = getListOfFiles(getConfig.getRawFixedFilesDir).map(_.getPath)

  val total_files_nums = math.min(buggy_files.size, fixed_files.size)

  val batch_size =total_files_nums / nums_worker


  val workers  = for (index <- 0 until nums_worker) yield {
    val start = index*batch_size
    val end = if (index == nums_worker - 1) total_files_nums else (index + 1) * batch_size

    new Worker( src_batch = buggy_files.slice(start, end),
      tgt_batch = fixed_files.slice(start, end),
      idioms=project_idioms,
      worker_id = index)
  }

  def readIdioms(filePath:String) = {
    var idioms = new mutable.HashSet[String]()
    try{
      val stream = Files.lines(Paths.get(filePath))
      idioms.++=(stream.collect(Collectors.toSet[String]()))
    } catch {
      case e:Exception => e.printStackTrace()
    }
    idioms
  }

  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  def data_preprocess() = {
    buggy_files.foreach(println _)
    fixed_files.foreach(println _)
  }



  def run() = {
    workers.foreach(_.call())
    data_preprocess
  }
}
