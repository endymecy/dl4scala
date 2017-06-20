package org.dl4scala.examples.recurrent.character.melodl4scala
import javax.sound.midi.{InvalidMidiDataException, MidiEvent, ShortMessage, Track}

/**
  * Created by endy on 2017/6/20.
  */
class Note(startTick: Long, rawNote: Int, velocity: Int, channel: Int) extends NoteOrInstrumentChange {
  private var durationInTicks = 0L // set later

  def getRawNote: Int = rawNote

  def getKey: Int = rawNote % 12

  def setDuration(durationInTicks: Long): Unit = {
    this.durationInTicks = durationInTicks
  }

  def getOctave: Int = rawNote / 12 - 1

  def getChannel: Int = channel

  def getVelocity: Int = velocity

  def interval(other: Note): Int = getRawNote - other.getRawNote

  def endTick: Long = startTick + durationInTicks

  override def toString: String = {
    val endTick = startTick + durationInTicks
    "Note[startTick: " + startTick + ", endTick = " + endTick + ", duration: " + durationInTicks + ", rawNote:" + rawNote +
      ", note: " + getKey + ", octave: " + getOctave + ", channel: " + channel + ", velocity: " + velocity + "] "
  }

  def getDuration: Long = durationInTicks


  @throws(classOf[InvalidMidiDataException])
  override def addMidiEvents(track: Track): Unit = {
    val midiMessageStart = new ShortMessage(ShortMessage.NOTE_ON, channel, rawNote, velocity)
    track.add(new MidiEvent(midiMessageStart, startTick))
    val midiMessageEnd = new ShortMessage(ShortMessage.NOTE_OFF, channel, rawNote, 0)
    track.add(new MidiEvent(midiMessageEnd, startTick + durationInTicks))
  }

  def getEndTick: Long = startTick + durationInTicks
}
