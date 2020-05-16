package org.ucf.ml
package parallel


import java.util.concurrent.Callable
import scala.collection.mutable

/**
 *
 * @param src_batch, a list of buggy (source) input files' path
 * @param tgt_batch, a list of fixed (target) input files' path
 * @param idioms, a bag of idioms vocabulary keep by this project
 * @param worker_id, integer id assigned by master
 */
class Worker(src_batch:List[String] = null,
             tgt_batch:List[String] = null,
             idioms:mutable.HashSet[String],
             worker_id:Int, granularity: Value = METHOD) extends Callable[String] with utils.Common{

  val javaPaser = new parser.JavaParser
  val ctx = new Context(idioms, granularity)

//  if (logger.isDebugEnabled){
//    ctx.setNewLine(true)
//    ctx.setIsAbstract(true)
//  }

  val batch_size = scala.math.min(src_batch.size, tgt_batch.size)

  def abstract_task(inputPath:String, mode:Value, granularity:Value = this.granularity) = {

    logger.debug(f"Worker ${worker_id} process ${mode} Source code ${inputPath}")
    ctx.setCurrentMode(mode)

    val cu = javaPaser.getComplationUnit(inputPath, granularity)

    javaPaser.addPositionWithGenCode(ctx, cu)

    if (logger.isDebugEnabled) {
      println(cu)
      println(ctx.get_abstract.split("\n").last)
      println("******************************************************\n")
    }
  }

  def task(buggyPath:String, fixedPath:String) = {

    abstract_task(buggyPath, SOURCE)
    abstract_task(fixedPath, TARGET)

    /*Dumpy buggy and fixed abstract code to a specify file*/

    /*Clear the context and */
    ctx.clear
  }

  def job() = {

    /*Iteration Executing task to handle with all involved in data*/
    for (idx <- 0 until batch_size) {
      task(src_batch(idx), tgt_batch(idx))
    }
    EmptyString
  }
  override def call(): String = job()
}
