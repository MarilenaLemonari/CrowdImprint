<div align="center">
<h1>CrowdImprint: Decomposing Context-Aware Interactions</h1>
<strong><a href="https://marilenalemonari.github.io/" target="_blank">Marilena Lemonari</a>, <a href="https://totis77.github.io/" target="_blank">Panayiotis Charalambous</a>, <a href="https://people.rennes.inria.fr/Julien.Pettre/" target="_blank">Julien Pettré</a>, and <a href="http://www.cs.ucy.ac.cy/~yiorgos/" target="_blank">Yiorgos Chrysanthou</a>

The Visual Computer, January 2026</strong></br>
</div>

![Demo Image](https://github.com/MarilenaLemonari/CrowdImprint/blob/main/Misc/Images/ci_teaser.png)

<p align="justify">
Crowd authoring has mainly focused on generalised agent interactions such as collision avoidance and grouping. However, in
society, people interact more intentionally with specific "sources" such as exhibits, or inspectors. Uncovering these interactions
is essential for understanding and characterising social behaviours. We propose a model that learns from trajectories, the
localised agent interactions imposed by the context of the object or agent source. Our model decomposes agent paths into
sequential combinations of simple and understandable "core" behaviours, like approach, stop, and circle around, temporally
dissecting source-centric trajectories into standardised movements. We train on pairs of trajectory-encoded images and their
associated core behaviour combination. Given a set of trajectories around a specific source, our framework can be applied to
build a behaviour distribution, summarising how people interact with the source type. The inferred distribution can then be
sampled to generate diverse crowds of context-aware agents. We evaluate our model using collected ground-truth data and
perform a case study that showcases the utility of this decomposition of context-aware interactions in other tasks, such as
measuring behaviour similarity.
</p>

<br>

<p align="center"><strong>
	- <a href="https://doi.org/10.1007/s00371-025-04329-2" target="_blank">Publication</a> | <a href="https://github.com/MarilenaLemonari/CrowdImprint/blob/main/Misc/PDF_files/CI_suppemetrary.pdf" target="_blank">Supplementary Material (PDF)</a> | <a href="https://youtu.be/SEhdstN5mgM" target="_blank">Video</a> -
</strong>
</p>

<br>

<p align="center" dir="auto">
	<a href="https://youtu.be/SEhdstN5mgM" rel="nofollow">
		<img align="center" width="600px" src="https://github.com/MarilenaLemonari/CrowdImprint/blob/main/Misc/Images/ci_youtube.png"/>
	</a>
</p>

### 📖 Citation
```bibtex
@article{Lemonari2026CrowdImprint,
  title={CrowdImprint: decomposing context-aware interactions},
  author={Lemonari, Marilena and Charalambous, Panayiotis and Pettr{\'e}, Julien and Chrysanthou, Yiorgos},
  journal={The Visual Computer},
  volume={42},
  number={1},
  pages={128},
  year={2026},
  publisher={Springer}
}
```
