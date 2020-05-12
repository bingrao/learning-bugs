package org.ucf.ml
package utils

import java.util.concurrent.atomic.AtomicInteger
class Context {
  private val position_offset = new AtomicInteger()
  def getNewPosition = position_offset.getAndIncrement()
  def getCurrentPositionOffset = position_offset.get()

  private var current_target = "buggy"
  def getCurrentTarget = this.current_target
  def setCurrentTarget(target:String) = this.current_target = target

  private val buggy_abstract = new StringBuilder()
  private val fixed_abstract = new StringBuilder()

//  def attachePosition(content:String) = f"${content}#${this.getNewPosition} "
  def attachePosition(content:String) = f"${content} "

  def append(content:String) = this.getCurrentTarget match {
    case "buggy" => this.buggy_abstract.append(attachePosition(content))
    case "fixed" => this.fixed_abstract.append(attachePosition(content))
    case _ =>
  }

  def get_buggy_abstract = buggy_abstract.toString()
  def get_fixed_abstract = fixed_abstract.toString()
}
