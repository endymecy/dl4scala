package org.dl4scala.examples.recurrent.character.melodl4scala
import javax.sound.midi.{InvalidMidiDataException, MidiEvent, ShortMessage, Track}

/**
  * Created by endy on 2017/6/20.
  */
class InstrumentChange(instrumentNumber: Int, tick: Long, channel: Int)
  extends NoteOrInstrumentChange{

  this.startTick = tick

  override def toString: String = "Change instrument to " + instrumentNumber +
    " (" + Midi2MelodyStrings.programs(instrumentNumber) + ") at " + startTick

  def getInstrumentNumber: Int = instrumentNumber

  @throws(classOf[InvalidMidiDataException])
  def addMidiEvents(track: Track): Unit = {
    val midiMessage = new ShortMessage(ShortMessage.PROGRAM_CHANGE, channel, instrumentNumber, 0)
    System.out.println("Adding instrument change to track for channel " + channel +
      " and instrumentName = " + Midi2MelodyStrings.programs(instrumentNumber))
    track.add(new MidiEvent(midiMessage, startTick))
  }
}
