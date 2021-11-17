//-- open the file outputted by File>Save>XY Coordinates

lineseparator = "\n";
cellseparator = ",\t";

// copies the whole RT to an array of lines
lines=split(File.openAsString("/Users/esthomas/Andor_Rotation/github_repo/cross-modal-autoencoders/roi_with_index.csv"), lineseparator);

//-- Clear the ROI manager
roiManager("reset");
//-- loop through the file
for (lineNum=1;lineNum<lines.length;lineNum++){
//-- Debug Only
//print(lines[lineNum]);
//-- extract the coordinates by splitting on tab
pointCoord=split(lines[lineNum],",\t");
//-- Draw the point
makePoint(pointCoord[1], pointCoord[2], "small yellow hybrid");
//-- add it to the ROI manager
roiManager("add");
}

run("Select None");