package org.dl4scala.examples.recurrent.character.melodl4scala

import javax.sound.midi.{InvalidMidiDataException, Track}

/**
  * Created by endy on 2017/6/20.
  */
abstract class NoteOrInstrumentChange {

  protected var startTick = 0L

  @throws(classOf[InvalidMidiDataException])
  abstract def addMidiEvents(track: Track): Unit
}
