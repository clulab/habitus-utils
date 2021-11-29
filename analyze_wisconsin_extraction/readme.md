
---------- Forwarded message ---------
From: Mihai Surdeanu <surdeanu@gmail.com>
Date: Sat, Nov 27, 2021 at 7:59 PM
Subject: [EXT]Re: [habitus] analyzing output of wisconsin runs
To: Mitch Paul Mithun <mithunpaul@email.arizona.edu>
Cc: Mihai Surdeanu <msurdeanu@email.arizona.edu>


External Email

Inline.

On Sat, Nov 27, 2021 at 7:14 PM Mitch Paul Mithun <mithunpaul@email.arizona.edu> wrote:
Mihai,

These are the stuff you wanted from the analysis of wisconsin runs. about which i have some questions:

Mithun: As soon as we get intermediate output from Wisc, please analyze it:
- How much output?
Qn) do you mean how many lines were there in the overall output file, say mentions.tsv? 

Yes, but given that this run had that duplicated bug, I'd say *unique* lines in the tsv file.
 

- How many have context, and what type of context?
Qn) can you define context? In the version of the code that was run by Ian, we print context (most frequent LOC, YEAR, CROP) for all event mentions,  (sample output pasted below).
I'd like to see stats about all:
- How many lines have LOC.
- How many lines have YEAR.
- How many lines have CROP.
- How many lines have all three.
(again, after removing duplicates)


- How many rows are about Senegal?
Qn) Did you mean how many rows had Senegal as their most frequent LOC in context?

How many rows have Senegal anywhere.
And, yes, Senegal as the most freq as well.
 

- Out of the Senegal rows, how many have the other context, i.e., year and/or cultivar?

Two stats:
- Out of the Senegal rows, how many lines have YEAR.
-  Out of the Senegal rows, how many lines have CROP.
- How many have both.