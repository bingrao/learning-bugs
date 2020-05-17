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
             worker_id:Int, granularity: Value = METHOD) extends Callable[Context] with utils.Common{

  val javaPaser = new parser.JavaParser
  val ctx = new Context(idioms, granularity)

  val batch_size = scala.math.min(src_batch.size, tgt_batch.size)

  def abstract_task(inputPath:String, mode:Value, granularity:Value = this.granularity) = {

    ctx.setCurrentMode(mode)

//    if (logger.isDebugEnabled) ctx.append(inputPath + "\t")

    val cu = javaPaser.getComplationUnit(inputPath, granularity)

    javaPaser.addPositionWithGenCode(ctx, cu)

//    if (logger.isDebugEnabled) {
//      logger.debug(f"Worker ${worker_id} process ${mode} Source code ${inputPath}")
//      println(cu)
//      println(ctx.get_abstract.split("\n").last)
//      println("******************************************************\n")
//    }
  }

  def task(buggyPath:String, fixedPath:String) = {

    abstract_task(buggyPath, SOURCE)
    abstract_task(fixedPath, TARGET)

    /*Dumpy buggy and fixed abstract code to a specify file*/

    /*Clear the context and */
    ctx.clear
  }

  def job(): Context = {
    val start = System.currentTimeMillis()
    /*Iteration Executing task to handle with all involved in data*/
    for (idx <- 0 until batch_size) {
      task(src_batch(idx), tgt_batch(idx))
    }
    val stop = System.currentTimeMillis()
    logger.info(f"Worker ${worker_id} deal with ${batch_size} task in ${stop - start} milliseconds")
    ctx
  }
  override def call(): Context = job()
}
